"""Independent wall-clock video senders with natural generation and per-lane metrics."""

import json
import math
import queue
import threading
import time

from semantic_checks import stats

PROTOCOL = "independent_realtime_v1"
QUESTION = "Describe what is happening in the video in detail."
FRAME_COUNT = 12
PROMPT_OFFSETS = (3.25, 7.25)
WARMUP_FRAMES = 2


def workload(case, fps):
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("FPS must be positive and finite")
    frames = [e for e in case["events"] if e["type"] == "frame"][:FRAME_COUNT]
    if len(frames) != FRAME_COUNT:
        raise ValueError("Expected 12 input frames")
    events = [
        dict(
            type="frame",
            offset=i / fps,
            timestamp=float(e["timestamp"]),
            frame_path=e["frame_path"],
            sha256=e["sha256"],
            frame_index=i,
        )
        for i, e in enumerate(frames)
    ]
    for index, offset in enumerate(PROMPT_OFFSETS):
        events.append(
            dict(
                type="prompt",
                offset=offset / fps,
                timestamp=offset,
                prompt=QUESTION,
                question_index=index,
            )
        )
    events.sort(key=lambda e: e["offset"])
    for seq, event in enumerate(events):
        event["seq_no"] = seq
        event["final"] = seq == len(events) - 1
    return events


def send_lane(engine, rid, lane, events, stagger, drain_seconds):
    try:
        if not lane["ready"].wait(60):
            raise TimeoutError("Session readiness timeout")
        if lane["cancel"].is_set():
            return
        epoch = time.perf_counter() + stagger
        lane["epoch"] = epoch
        for event in events:
            due = epoch + event["offset"]
            if lane["cancel"].wait(max(0, due - time.perf_counter())):
                return
            sent = time.perf_counter()
            record = dict(event, planned_at=due, sent_at=sent)
            with lane["lock"]:
                lane["sent"].append(record)
            engine.update(rid, event, event["seq_no"], final=event["final"])
        lane["drain_deadline"] = time.perf_counter() + drain_seconds
    except BaseException as exc:
        with lane["lock"]:
            lane["errors"].append(f"{type(exc).__name__}: {exc}")
    finally:
        lane["sender_done"].set()


def lane_result(index, lane, records, special_ids):
    sent, received = lane["sent"], lane["received"]
    errors = list(lane["errors"])
    processed = {}
    for event in received:
        if event.get("event") in ("input.frame.processed", "input.prompt.processed"):
            seq = event["seq_no"]
            if seq in processed:
                errors.append(f"Duplicate processed event: {seq}")
            processed[seq] = event
    sent_ids = {e["seq_no"] for e in sent}
    missing = sorted(sent_ids - set(processed))
    unexpected = sorted(set(processed) - sent_ids)
    if missing or unexpected:
        errors.append(f"Input mismatch: missing={missing}, unexpected={unexpected}")
    frames, questions = [], []
    for item in sent:
        done = processed.get(item["seq_no"])
        if item["type"] == "frame":
            frames.append(
                dict(
                    seq_no=item["seq_no"],
                    frame_index=item["frame_index"],
                    sha256=item["sha256"],
                    planned_at=item["planned_at"],
                    sent_at=item["sent_at"],
                    processed_at=done["received_at"] if done else None,
                )
            )
        else:
            turn = done.get("turn_id") if done else None
            text_events = [
                e
                for e in received
                if e.get("modality") == "text"
                and e.get("turn_id") == turn
                and isinstance(e.get("text"), str)
            ]
            first = min(
                (e["received_at"] for e in text_events if e["text"].strip()),
                default=None,
            )
            questions.append(
                dict(
                    seq_no=item["seq_no"],
                    turn_id=turn,
                    planned_at=item["planned_at"],
                    sent_at=item["sent_at"],
                    processed_at=done["received_at"] if done else None,
                    first_text_at=first,
                    text="".join(e["text"] for e in text_events),
                )
            )
            if first is None:
                errors.append(f"No visible answer for prompt {item['seq_no']}")
    turns = {q["turn_id"] for q in questions if q["turn_id"] is not None}
    intervals, generated, batches = [], 0, []
    last = None
    for record in records:
        if record["turn_id"] not in turns or record["token"] in special_ids:
            last = None
            continue
        generated += 1
        batches.append(record["batch_size"])
        if last is not None and last["turn_id"] == record["turn_id"]:
            intervals.append(record["time"] - last["time"])
        last = record
    if not intervals:
        errors.append("No within-answer token intervals")
    return dict(
        lane=index,
        epoch=lane.get("epoch"),
        ended_at=lane["ended_at"],
        frames=frames,
        questions=questions,
        decode_intervals=intervals,
        generated_tokens=generated,
        decode_batch_sizes=batches,
        peak_pending_events=lane["peak_pending"],
        errors=errors,
        missing_events=missing,
        unexpected_events=unexpected,
        scheduled_events=len(lane["schedule"]),
        sent_events=len(sent),
        processed_events=len(processed),
        received=received,
        token_records=records,
    )


def measure(engine, case, count, *, fps=1.0, token_rate=10.0, drain_seconds=15.0):
    events = workload(case, fps)
    initial = dict(case)
    initial["initial_prompt"] = (
        "Watch the video. Stay silent until a question is asked."
    )
    states, threads, rids = {}, [], []
    special = set(engine.tokenizer.all_special_ids) | {engine.silence}
    result = []
    try:
        for index in range(count):
            rid = engine.new(
                initial,
                force=None,
                allowance=4096,
                benchmark=False,
                token_rate=token_rate,
            )
            rids.append(rid)
            lane = dict(
                index=index,
                ready=threading.Event(),
                cancel=threading.Event(),
                sender_done=threading.Event(),
                lock=threading.Lock(),
                sent=[],
                received=[],
                errors=[],
                schedule=events,
                peak_pending=0,
                ended_at=None,
                created_at=time.perf_counter(),
            )
            states[rid] = lane
            thread = threading.Thread(
                target=send_lane,
                args=(engine, rid, lane, events, index * 0.05 / fps, drain_seconds),
                daemon=True,
            )
            threads.append(thread)
            thread.start()
        active = set(rids)
        while active:
            now = time.perf_counter()
            for rid in list(active):
                lane = states[rid]
                deadline = lane.get(
                    "drain_deadline",
                    lane["created_at"]
                    + 60
                    + events[-1]["offset"]
                    + drain_seconds
                    + count * 0.05 / fps,
                )
                if lane["errors"] or now > deadline:
                    if now > deadline:
                        lane["errors"].append("Session drain timeout")
                    lane["ended_at"] = now
                    lane["cancel"].set()
                    lane["ready"].set()
                    active.remove(rid)
                    engine.scheduler.abort(rid)
            if not active:
                break
            try:
                message = engine.scheduler.outbox.get(timeout=0.02)
            except queue.Empty:
                if not engine.thread.is_alive():
                    raise RuntimeError(engine.thread_error or "Scheduler stopped")
                continue
            rid = message.request_id
            if rid not in active:
                continue
            lane = states[rid]
            arrived = time.perf_counter()
            data = dict(message.data) if isinstance(message.data, dict) else {}
            data.update(received_at=arrived, message_type=message.type)
            with lane["lock"]:
                lane["received"].append(data)
                processed = {
                    e["seq_no"]
                    for e in lane["received"]
                    if e.get("event")
                    in ("input.frame.processed", "input.prompt.processed")
                }
                # Include this just-completed event in the queue high-water mark.
                pending_before = (
                    len(lane["sent"])
                    - len(processed)
                    + int(
                        data.get("event")
                        in ("input.frame.processed", "input.prompt.processed")
                    )
                )
                lane["peak_pending"] = max(lane["peak_pending"], pending_before)
            if data.get("event") == "session.ready":
                lane["ready"].set()
            if message.type in ("error", "result"):
                if message.type == "error":
                    lane["errors"].append(str(message.data))
                if len(lane["sent"]) != len(events):
                    lane["errors"].append("Session ended before all scheduled inputs")
                lane["ended_at"] = arrived
                lane["cancel"].set()
                lane["ready"].set()
                active.remove(rid)
        for thread in threads:
            thread.join(2)
            if thread.is_alive():
                raise RuntimeError("Input sender did not stop")
        for index, rid in enumerate(rids):
            result.append(
                lane_result(
                    index, states[rid], list(engine.streams[rid]["records"]), special
                )
            )
    finally:
        for state in states.values():
            state["cancel"].set()
            state["ready"].set()
        for thread in threads:
            thread.join(2)
        if not engine.recover(rids):
            raise RuntimeError("Session KV not recovered")
    return dict(
        protocol=PROTOCOL,
        sessions=count,
        kv_recovered=True,
        fps=fps,
        token_rate=token_rate,
        lanes=result,
    )


def measurement_order(sessions, repeats):
    for trial in range(repeats):
        offset = trial % len(sessions)
        for count in sessions[offset:] + sessions[:offset]:
            yield trial, count


def run_interleaved(engine, case, sessions, repeats, save, *, fps=1.0, token_rate=10.0):
    rows = []
    for trial, count in measurement_order(sessions, repeats):
        current = measure(engine, case, count, fps=fps, token_rate=token_rate)
        current["trial"] = trial
        rows.append(current)
        save(rows)
        print(
            "INDEPENDENT",
            count,
            "trial",
            trial + 1,
            "/",
            repeats,
            "errors",
            sum(bool(l["errors"]) for l in current["lanes"]),
            flush=True,
        )
    return rows


def summarize(trials, sessions, repeats):
    expected = {(n, r) for n in sessions for r in range(repeats)}
    if (
        len(trials) != len(expected)
        or {(r["sessions"], r["trial"]) for r in trials} != expected
    ):
        raise ValueError("Incomplete or duplicate concurrency trials")
    signature = None
    for trial in trials:
        n, lanes = trial["sessions"], trial["lanes"]
        if (
            trial.get("protocol") != PROTOCOL
            or not trial["kv_recovered"]
            or len(lanes) != n
            or {l["lane"] for l in lanes} != set(range(n))
        ):
            raise ValueError("Protocol, lane or KV validation failed")
        if any(l["epoch"] is None or l["ended_at"] is None for l in lanes):
            raise ValueError("Incomplete session timeline")
        if max(l["epoch"] for l in lanes) >= min(l["ended_at"] for l in lanes):
            raise ValueError("Sessions did not overlap")
        for lane in lanes:
            hashes = [f["sha256"] for f in lane["frames"]]
            if signature is None:
                signature = hashes
            if hashes != signature or len(hashes) != FRAME_COUNT:
                raise ValueError("Frame workload mismatch")
    rows = []
    for n in sorted(sessions):
        selected = [t for t in trials if t["sessions"] == n]
        all_lanes = [l for t in selected for l in t["lanes"]]

        def metrics(lanes):
            frame_delays = [
                f["processed_at"] - f["sent_at"]
                for l in lanes
                for f in l["frames"]
                if f["frame_index"] >= WARMUP_FRAMES and f["processed_at"] is not None
            ]
            ttft = [
                q["first_text_at"] - q["sent_at"]
                for l in lanes
                for q in l["questions"]
                if q["first_text_at"] is not None
            ]
            lag = [
                x["sent_at"] - x["planned_at"]
                for l in lanes
                for x in l["frames"] + l["questions"]
            ]
            gaps = [t for l in lanes for t in l["decode_intervals"]]
            if any(
                not math.isfinite(t) or t < 0 for t in frame_delays + ttft + lag + gaps
            ):
                raise ValueError("Invalid timing sample")
            return dict(
                frame=stats(frame_delays),
                ttft=stats(ttft),
                tpot=stats(gaps),
                send_lag=stats(lag),
                active_tokens_per_second=(
                    len(gaps) / sum(gaps) if gaps and sum(gaps) > 0 else None
                ),
                generated_tokens=sum(l["generated_tokens"] for l in lanes),
                missing_frames=sum(
                    f["processed_at"] is None for l in lanes for f in l["frames"]
                ),
                unanswered_prompts=sum(
                    q["first_text_at"] is None for l in lanes for q in l["questions"]
                ),
                failed_lanes=sum(bool(l["errors"]) for l in lanes),
                peak_pending_events=max(l["peak_pending_events"] for l in lanes),
            )

        row = dict(sessions=n, **metrics(all_lanes))
        row["lanes"] = [
            dict(lane=i, **metrics([l for l in all_lanes if l["lane"] == i]))
            for i in range(n)
        ]
        rates = [
            l["active_tokens_per_second"]
            for l in row["lanes"]
            if l["active_tokens_per_second"] is not None
        ]
        row["mean_lane_tokens_per_second"] = sum(rates) / len(rates) if rates else None
        row["slowest_lane_tokens_per_second"] = min(rates) if rates else None
        wall = sum(
            max(l["ended_at"] for l in t["lanes"]) - min(l["epoch"] for l in t["lanes"])
            for t in selected
        )
        row["aggregate_wall_tokens_per_second"] = row["generated_tokens"] / wall
        row["batch_histogram"] = {
            str(b): sum(l["decode_batch_sizes"].count(b) for l in all_lanes)
            for b in range(1, n + 1)
        }
        rows.append(row)
    base = next(r for r in rows if r["sessions"] == 1)["tpot"]["mean"]
    for row in rows:
        row["tpot_ratio"] = (
            row["tpot"]["mean"] / base
            if base and row["tpot"]["mean"] is not None
            else None
        )
    return rows


def number(value, scale=1):
    return "N/A" if value is None else f"{value*scale:.3f}"


def report(output, metadata, errors):
    raw, rows = [], []
    try:
        raw = json.loads((output / "sglang_concurrency.json").read_text())
        rows = summarize(raw, metadata["sessions"], metadata["repeats"])
        if any(
            r["failed_lanes"] or r["unanswered_prompts"] or r["missing_frames"]
            for r in rows
        ):
            errors.append(
                "Some sessions have missing inputs, unanswered prompts or execution errors"
            )
    except (OSError, ValueError, KeyError, TypeError, IndexError, StopIteration) as exc:
        errors.append("Independent concurrency validation: " + str(exc))
    lines = [
        "# SGLang 独立会话时延",
        "",
        f"每路独立按 {metadata.get('concurrency_fps',1)} FPS 发送 12 帧，在第 3.25/7.25 帧间隔处提问。",
        f"每路 token_rate={metadata.get('concurrency_token_rate',10)} token/s；该配置是调度速率目标，不保证每路实际速度达到此值。",
        "各路只等待自身 session.ready；发送不等待帧处理、silence、回答结束或其他会话。",
        "自然生成，不强制 token；前两帧作为输入预热，不计入 frame 统计。",
        "TTFT 为提问至首段可见文本；TPOT 统计同一连续回答的普通 token 间隔，包含期间帧处理造成的停顿，排除静默期。",
        "进程内真实调度，不含 WebSocket/网关/网络；未响应问题、未处理帧和发送延迟单独记录。",
        "单路与多路使用本次同协议基线；不复用旧同步屏障或固定 token 基准。正式服务仍为 4 slots。",
        "",
        "| Sessions | Frame mean / P95 ms | TTFT mean / P95 ms | TPOT mean / P95 ms | TPOT 相对单路 | 每路 active token/s | 最慢一路 active token/s | 整体 wall token/s | 未处理帧 / 未回答问题 / 异常会话 |",
        "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for r in rows:
        values = [str(r["sessions"])]
        values += [
            number(r[k]["mean"], 1000) + " / " + number(r[k]["p95"], 1000)
            for k in ("frame", "ttft", "tpot")
        ]
        values += [
            number(r["tpot_ratio"]),
            number(r["mean_lane_tokens_per_second"]),
            number(r["slowest_lane_tokens_per_second"]),
            number(r["aggregate_wall_tokens_per_second"]),
            f"{r['missing_frames']} / {r['unanswered_prompts']} / {r['failed_lanes']}",
        ]
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "## 逐路结果",
            "",
            "| Sessions / Lane | Frame P95 ms | TTFT mean / P95 ms | TPOT mean / P95 ms | active token/s | 发送迟到 P95 ms | 最大待处理事件 |",
            "| --- | ---: | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for r in rows:
        for l in r["lanes"]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"{r['sessions']} / {l['lane']}",
                        number(l["frame"]["p95"], 1000),
                        number(l["ttft"]["mean"], 1000)
                        + " / "
                        + number(l["ttft"]["p95"], 1000),
                        number(l["tpot"]["mean"], 1000)
                        + " / "
                        + number(l["tpot"]["p95"], 1000),
                        number(l["active_tokens_per_second"]),
                        number(l["send_lag"]["p95"], 1000),
                        str(l["peak_pending_events"]),
                    ]
                )
                + " |"
            )
    return dict(protocol=PROTOCOL, concurrency=rows, results=raw), lines
