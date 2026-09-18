"""Protocol smoke client for the legacy VL adapter (VL-01/02/03).

Implements the model-side self-test items from the requirements doc §6.1:

* VL-01: start, then a batch of 4 and a batch of 8 JPEGs; expects ready,
  an equal number of frame_ack, joinable incremental output text and the
  <|im_end|> end marker.
* VL-02: stop, disconnect and immediately start the next round; expects the
  session to be released (no busy residue) and rounds not to contaminate
  each other (round-2 output must not contain round-1's full text).
* VL-03: corrupt JPEG, busy-state message, abrupt disconnect mid-round, and
  a 1-frame minimal round exercising the 1s-silence finalization path;
  expects consumable errors and clean resource release.

Additionally VL-01 asserts that output text arrives as increments: no
message may contain the entire text accumulated so far (cumulative resend).

Run on the omni host (or anywhere that can reach the adapter):

    .venv-main/bin/python -m tests.smoke_client            # from vl_legacy_adapter/
    .venv-main/bin/python tests/smoke_client.py --url ws://127.0.0.1:18600/v1/realtime?session_id=

Every check prints PASS/FAIL and a JSON timing record per round.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import io
import json
import os
import sys
import time

from PIL import Image
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

CONNECT_OPTIONS = (
    {"proxy": None} if "proxy" in inspect.signature(connect).parameters else {}
)

END_MARKERS = (
    "<|im_end|>",
    "<|silence|>",
    "<|round_end|>",
    "<|eot_id|>",
    "<|endoftext|>",
)
BUSY_SUBSTRING = "realtime session is already active"

READY_TIMEOUT_S = 10.0
ROUND_TIMEOUT_S = 10.0
BUSY_RETRY_DELAYS = (0.5, 1.0, 1.5)

# The §5.2 example start message, verbatim.
START_EXAMPLE = {
    "type": "start",
    "prompt": "请描述这些画面中正在发生的事情。",
    "frame_queue_size": 32,
    "max_new_tokens": 512,
    "max_tokens_per_second": 160,
    "do_sample": False,
    "temperature": 0.2,
    "top_k": 20,
    "top_p": 0.8,
    "repetition_penalty": 1.05,
}


def load_jpegs(testdata_dir: str, count: int) -> list[bytes]:
    names = sorted(n for n in os.listdir(testdata_dir) if n.endswith(".png"))
    if len(names) < count:
        raise RuntimeError(f"need {count} frames in {testdata_dir}, found {len(names)}")
    out = []
    for name in names[:count]:
        with Image.open(os.path.join(testdata_dir, name)) as im:
            buf = io.BytesIO()
            im.convert("RGB").save(buf, format="JPEG", quality=90)
            out.append(buf.getvalue())
    return out


def synth_jpegs(count: int) -> list[bytes]:
    out = []
    for i in range(count):
        im = Image.new("RGB", (640, 360), color=((i * 37) % 255, 96, 160))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=85)
        out.append(buf.getvalue())
    return out


class RoundResult:
    def __init__(self) -> None:
        self.acks = 0
        self.outputs: list[str] = []
        self.errors: list[str] = []
        self.retry_errors: list[str] = []
        self.marker_seen = False
        self.t_start = 0.0
        self.t_ready: float | None = None
        self.t_first_ack: float | None = None
        self.t_last_ack: float | None = None
        self.t_first_output: float | None = None
        self.t_marker: float | None = None
        self.t_total: float | None = None

    @property
    def full_text(self) -> str:
        return "".join(self.outputs)

    @property
    def visible_text(self) -> str:
        text = self.full_text
        for marker in END_MARKERS:
            text = text.replace(marker, "")
        return text

    def timings(self) -> dict:
        def rel(t: float | None) -> float | None:
            return round(t - self.t_start, 3) if t is not None else None

        return {
            "ready_s": rel(self.t_ready),
            "first_ack_s": rel(self.t_first_ack),
            "last_ack_s": rel(self.t_last_ack),
            "first_output_s": rel(self.t_first_output),
            "marker_s": rel(self.t_marker),
            "total_s": round(self.t_total, 3) if self.t_total else None,
        }


def outputs_are_incremental(outputs: list[str]) -> bool:
    """Each output must be a delta, not a cumulative resend: no message may
    contain the entire text accumulated from all previous messages."""
    accumulated = ""
    for text in outputs:
        if accumulated and accumulated in text:
            return False
        accumulated += text
    return True


async def recv_json(ws, timeout: float) -> dict | None:
    raw = await asyncio.wait_for(ws.recv(), timeout)
    if isinstance(raw, bytes):
        raise RuntimeError("unexpected binary message from adapter")
    return json.loads(raw)


async def run_round(
    url: str,
    frames: list[bytes],
    *,
    prompt: str | None = None,
    max_new_tokens: int | None = None,
    retry_busy: bool = True,
) -> RoundResult:
    """Retry rejected connections only, within one shared round deadline."""
    started = time.monotonic()
    deadline = started + ROUND_TIMEOUT_S
    retries = []
    result = RoundResult()
    delays = BUSY_RETRY_DELAYS if retry_busy else ()
    try:
        async with asyncio.timeout_at(deadline):
            for attempt in range(len(delays) + 1):
                result = await _run_round_once(
                    url,
                    frames,
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    deadline=deadline,
                )
                busy = (
                    result.t_ready is None
                    and len(result.errors) == 1
                    and BUSY_SUBSTRING in result.errors[0]
                )
                if not busy or attempt == len(delays):
                    break
                retries.extend(result.errors)
                await asyncio.sleep(delays[attempt])
    except (TimeoutError, ConnectionClosed, OSError) as exc:
        result = RoundResult()
        result.errors.append(f"round failed: {exc!r}")
    result.retry_errors = retries
    result.t_start = started
    result.t_total = time.monotonic() - started
    return result


async def _run_round_once(
    url: str,
    frames: list[bytes],
    *,
    prompt: str | None = None,
    max_new_tokens: int | None = None,
    deadline: float,
) -> RoundResult:
    """One full legacy-contract round: start -> ready -> frames -> outputs -> stop."""
    result = RoundResult()
    result.t_start = time.monotonic()
    async with connect(
        url, **CONNECT_OPTIONS, max_size=16 * 1024 * 1024, open_timeout=READY_TIMEOUT_S
    ) as ws:
        start = dict(START_EXAMPLE)
        if prompt is not None:
            start["prompt"] = prompt
        if max_new_tokens is not None:
            start["max_new_tokens"] = max_new_tokens
        await ws.send(json.dumps(start, ensure_ascii=False))
        # Wait for ready.
        while True:
            message = await recv_json(ws, max(0.1, deadline - time.monotonic()))
            if message["type"] == "ready":
                result.t_ready = time.monotonic()
                break
            if message["type"] == "error":
                result.errors.append(message.get("message", ""))
                result.t_total = time.monotonic() - result.t_start
                return result
        # Batch-send all frames without waiting for individual acks.
        for index, payload in enumerate(frames):
            await ws.send(json.dumps({"type": "frame", "timestamp": float(index + 1)}))
            await ws.send(payload)
        # Collect acks and outputs until the end marker or an error.
        while True:
            try:
                message = await recv_json(ws, max(0.1, deadline - time.monotonic()))
            except (asyncio.TimeoutError, ConnectionClosed) as exc:
                result.errors.append(f"round ended without marker: {exc!r}")
                break
            mtype = message["type"]
            now = time.monotonic()
            if mtype == "frame_ack":
                result.acks += 1
                if result.t_first_ack is None:
                    result.t_first_ack = now
                result.t_last_ack = now
            elif mtype == "output":
                if result.t_first_output is None:
                    result.t_first_output = now
                text = message.get("text", "")
                result.outputs.append(text)
                if any(marker in text for marker in END_MARKERS):
                    result.marker_seen = True
                    result.t_marker = now
                    break
            elif mtype == "error":
                result.errors.append(message.get("message", ""))
                break
        try:
            await ws.send(json.dumps({"type": "stop"}))
        except ConnectionClosed:
            pass
    result.t_total = time.monotonic() - result.t_start
    return result


def report(name: str, ok: bool, detail: str, result: RoundResult | None = None) -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if result is not None:
        record = {
            "test": name,
            **result.timings(),
            "acks": result.acks,
            "text": result.visible_text[:80],
            "errors": result.errors,
            "retry_errors": result.retry_errors,
        }
        print("      " + json.dumps(record, ensure_ascii=False))
    return ok


async def test_vl01(url: str, frames: list[bytes]) -> bool:
    ok = True
    for count in (4, 8):
        result = await run_round(url, frames[:count])
        incremental = outputs_are_incremental(result.outputs)
        good = (
            result.t_ready is not None
            and result.acks == count
            and result.marker_seen
            and bool(result.visible_text.strip())
            and incremental
            and not result.errors
        )
        detail = (
            f"{count} JPEGs, acks={result.acks}, marker={result.marker_seen}, "
            f"visible={len(result.visible_text)} chars, incremental={incremental}"
        )
        ok &= report(f"VL-01/{count}", good, detail, result)
    return ok


async def test_vl02(url: str, frames: list[bytes]) -> bool:
    r1 = await run_round(
        url, frames[:4], prompt="第一轮：请描述画面中的主体。", retry_busy=False
    )
    r2 = await run_round(
        url, frames[:4], prompt="第二轮：画面里有什么动作？", retry_busy=False
    )
    good = (
        r1.marker_seen
        and r2.marker_seen
        and not any(BUSY_SUBSTRING in e for e in r2.errors)
        and not r1.errors
        and not r2.errors
    )
    ok = report(
        "VL-02", good, "two consecutive rounds, no busy residue, no cross-talk", r2
    )
    print(f"      round1 text: {r1.visible_text[:60]!r}")
    print(f"      round2 text: {r2.visible_text[:60]!r}")
    # Cross-round contamination: round 2 must not replay round 1's answer.
    r1_text = r1.visible_text.strip()
    if len(r1_text) >= 8:
        leaked = r1_text in r2.full_text
        if leaked:
            ok = False
        ok &= report(
            "VL-02/no-carryover",
            not leaked,
            "round1 full text "
            + ("FOUND in" if leaked else "absent from")
            + " round2 output",
        )
    else:
        print(
            "      [MANUAL-CHECK] round1 produced too little visible text to "
            "auto-check carryover; verify by eye that round2 does not "
            "replay round1 output"
        )
    return ok


async def test_vl03_bad_jpeg(url: str, frames: list[bytes]) -> bool:
    corrupt = b"\xff\xd8\xff\xe0" + os.urandom(4096)
    result = await run_round(url, [frames[0], corrupt, frames[1]])
    consumable = bool(result.errors) or result.marker_seen
    ok = report(
        "VL-03/bad-jpeg",
        consumable,
        f"corrupt frame -> errors={result.errors} marker={result.marker_seen}",
        result,
    )
    # Resource cleanup: the next round must work.
    nxt = await run_round(url, frames[:4])
    ok &= report(
        "VL-03/bad-jpeg+next",
        nxt.marker_seen and not nxt.errors,
        "next round healthy after corrupt frame",
        nxt,
    )
    return ok


async def test_vl03_busy(url: str, frames: list[bytes]) -> bool:
    # Hold one session open (ready, frames unsent) and start another.
    async with connect(url, **CONNECT_OPTIONS, max_size=16 * 1024 * 1024) as ws1:
        await ws1.send(json.dumps(dict(START_EXAMPLE), ensure_ascii=False))
        first = await recv_json(ws1, READY_TIMEOUT_S)
        assert first["type"] == "ready", first
        busy_seen = False
        try:
            async with connect(
                url, **CONNECT_OPTIONS, max_size=16 * 1024 * 1024
            ) as ws2:
                await ws2.send(json.dumps(dict(START_EXAMPLE), ensure_ascii=False))
                message = await recv_json(ws2, READY_TIMEOUT_S)
                busy_seen = message[
                    "type"
                ] == "error" and BUSY_SUBSTRING in message.get("message", "")
        except ConnectionClosed as exc:
            busy_seen = False
            print(f"      second connection closed: {exc}")
        await ws1.send(json.dumps({"type": "stop"}))
    ok = report("VL-03/busy", busy_seen, "second session got the legacy busy message")
    await asyncio.sleep(0.3)
    nxt = await run_round(url, frames[:4])
    ok &= report(
        "VL-03/busy+next",
        nxt.marker_seen and not nxt.errors,
        "slot released after stop",
        nxt,
    )
    return ok


async def test_vl03_silent_round(url: str, frames: list[bytes]) -> bool:
    """1-frame minimal round: exercises the 1s-silence finalization path.

    If the model answers, the round must end via the forged <|im_end|> after
    ~1s of output silence; if the model stays silent the whole round, the
    adapter must deliver the "no visible output" error instead of a bare
    marker. Either is a clean round end; hanging or any other error is not.
    """
    result = await run_round(url, frames[:1], prompt="看一眼。", max_new_tokens=128)
    no_output_error = any("no visible output" in e for e in result.errors)
    other_errors = [e for e in result.errors if "no visible output" not in e]
    clean_end = (result.marker_seen and not result.errors) or (
        no_output_error and not other_errors
    )
    within_budget = result.t_total is not None and result.t_total < ROUND_TIMEOUT_S
    ok = report(
        "VL-03/silent-round",
        clean_end and within_budget,
        f"1-frame minimal round: marker={result.marker_seen} "
        f"errors={result.errors}",
        result,
    )
    if not result.marker_seen and not no_output_error:
        print(
            "      [MANUAL-CHECK] round ended neither via the forged marker "
            "nor the no-visible-output error; inspect adapter logs"
        )
    nxt = await run_round(url, frames[:4])
    ok &= report(
        "VL-03/silent+next",
        nxt.marker_seen and not nxt.errors,
        "next round healthy after silent round",
        nxt,
    )
    return ok


async def test_vl03_abrupt_close(url: str, frames: list[bytes]) -> bool:
    # Mid-round abrupt disconnect (no stop), then a fresh round must work.
    ws = await connect(url, **CONNECT_OPTIONS, max_size=16 * 1024 * 1024)
    await ws.send(json.dumps(dict(START_EXAMPLE), ensure_ascii=False))
    first = await recv_json(ws, READY_TIMEOUT_S)
    assert first["type"] == "ready", first
    for index, payload in enumerate(frames[:2]):
        await ws.send(json.dumps({"type": "frame", "timestamp": float(index + 1)}))
        await ws.send(payload)
    await asyncio.sleep(0.5)
    await ws.close()
    await asyncio.sleep(0.5)
    nxt = await run_round(url, frames[:4])
    return report(
        "VL-03/abrupt-close",
        nxt.marker_seen and not nxt.errors,
        "fresh round healthy after abrupt disconnect",
        nxt,
    )


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://127.0.0.1:18600/v1/realtime?session_id=")
    parser.add_argument(
        "--testdata",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "..",
            "..",
            "testdata",
            "moss_vl_realtime_1fps",
            "cd067_ovorec_L2_slow1fps_000013",
        ),
    )
    parser.add_argument("--tests", default="vl01,vl02,vl03")
    args = parser.parse_args()

    if os.path.isdir(args.testdata):
        frames = load_jpegs(args.testdata, 8)
        print(f"loaded 8 JPEG frames from {args.testdata}")
    else:
        frames = synth_jpegs(8)
        print("testdata not found, using synthetic JPEG frames")

    selected = {t.strip() for t in args.tests.split(",")}
    results: dict[str, bool] = {}
    if "vl01" in selected:
        results["VL-01"] = await test_vl01(args.url, frames)
    if "vl02" in selected:
        results["VL-02"] = await test_vl02(args.url, frames)
    if "vl03" in selected:
        results["VL-03"] = await test_vl03_bad_jpeg(args.url, frames)
        results["VL-03"] &= await test_vl03_busy(args.url, frames)
        results["VL-03"] &= await test_vl03_silent_round(args.url, frames)
        results["VL-03"] &= await test_vl03_abrupt_close(args.url, frames)

    print("\n==== summary ====")
    for name, ok in results.items():
        print(f"{name}: {'PASS' if ok else 'FAIL'}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
