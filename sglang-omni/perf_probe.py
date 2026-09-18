"""Legacy VL latency: connection start to first visible text and round completion."""

from __future__ import annotations

import argparse
import asyncio
import inspect
import io
import json
import math
import time
from contextlib import suppress
from pathlib import Path

from PIL import Image
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

CONNECT_OPTIONS = (
    {"proxy": None} if "proxy" in inspect.signature(connect).parameters else {}
)

ROOT = Path(__file__).resolve().parent
DEFAULT_CASE = ROOT / "deployment/moss_vl_realtime/cases/cd067_sbpro_L2_stream_000122"
END_MARKERS = (
    "<|im_end|>",
    "<|silence|>",
    "<|round_end|>",
    "<|eot_id|>",
    "<|endoftext|>",
)


def visible_text(text: str) -> str:
    for marker in END_MARKERS:
        text = text.replace(marker, "")
    # Do not mistake the incomplete prefix of a split terminal marker for text.
    for marker in END_MARKERS:
        for size in range(min(len(text), len(marker) - 1), 0, -1):
            if text.endswith(marker[:size]):
                text = text[:-size]
                break
    return text


async def one_round(url, frames, prompt, sampling, *, timeout_s=10.0):
    start_at = time.perf_counter()
    reader = None
    async with asyncio.timeout(timeout_s):
        async with connect(
            url,
            **CONNECT_OPTIONS,
            max_size=32 << 20,
            open_timeout=timeout_s,
            close_timeout=2,
        ) as ws:
            connect_s = time.perf_counter() - start_at
            await ws.send(
                json.dumps(
                    dict(
                        type="start",
                        prompt=prompt,
                        frame_queue_size=32,
                        max_new_tokens=sampling.get("max_new_tokens", 512),
                        max_tokens_per_second=sampling.get("token_rate", 160),
                        do_sample=False,
                        temperature=0.2,
                        top_k=20,
                        top_p=0.8,
                        repetition_penalty=1.05,
                    ),
                    ensure_ascii=False,
                )
            )
            try:
                message = json.loads(await ws.recv())
                if message.get("type") != "ready":
                    raise RuntimeError(f"expected ready: {message}")
                ready_s = time.perf_counter() - start_at

                async def collect():
                    acks, text, ttft_s, acks_s, ended = 0, "", None, None, False
                    while True:
                        message = json.loads(await ws.recv())
                        kind = message.get("type")
                        if kind == "error":
                            raise RuntimeError(message.get("message", "server error"))
                        if kind == "frame_ack":
                            acks += 1
                            if acks > len(frames):
                                raise RuntimeError(
                                    "received more frame ACKs than sent frames"
                                )
                            if acks == len(frames):
                                acks_s = time.perf_counter() - start_at
                        elif kind == "output":
                            if ended:
                                raise RuntimeError(
                                    "output arrived after the end marker"
                                )
                            delta = message.get("text")
                            if not isinstance(delta, str):
                                raise RuntimeError("output.text must be a string")
                            text += delta
                            if visible_text(text).strip() and ttft_s is None:
                                ttft_s = time.perf_counter() - start_at
                            ended = any(marker in text for marker in END_MARKERS)
                            if ended and not visible_text(text).strip():
                                raise RuntimeError("round ended without visible text")
                        else:
                            raise RuntimeError(f"unexpected server event: {message}")
                        if ended and acks == len(frames):
                            return dict(
                                connect=connect_s,
                                ready=ready_s,
                                acks=acks_s,
                                ttft=ttft_s,
                                total=time.perf_counter() - start_at,
                                text=visible_text(text),
                                end_reason="end_marker",
                                ack_count=acks,
                                chars=len(visible_text(text)),
                            )

                reader = asyncio.create_task(collect())
                for index, data in enumerate(frames):
                    if reader.done():
                        await reader
                        raise RuntimeError("round ended before all frames were sent")
                    await ws.send(
                        json.dumps(dict(type="frame", timestamp=float(index)))
                    )
                    await ws.send(data)
                return await reader
            finally:
                if reader is not None:
                    reader.cancel()
                    await asyncio.gather(reader, return_exceptions=True)
                with suppress(OSError, ConnectionClosed, TimeoutError):
                    async with asyncio.timeout(1):
                        await ws.send('{"type":"stop"}')


def load_frames(args):
    if args.image:
        paths = [args.image] * args.frames
    else:
        paths = sorted(
            path
            for path in args.frames_dir.iterdir()
            if path.suffix.lower() in (".jpg", ".jpeg", ".png")
        )[: args.frames]
        if len(paths) != args.frames:
            raise ValueError(f"need {args.frames} images in {args.frames_dir}")
    frames = []
    for path in paths:
        buffer = io.BytesIO()
        with Image.open(path) as image:
            image.convert("RGB").save(buffer, format="JPEG", quality=90)
        frames.append(buffer.getvalue())
    return frames


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://127.0.0.1:18600/v1/realtime")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--frames", type=int, default=4)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--image", type=Path, help="repeat this one image for all frames"
    )
    source.add_argument("--frames-dir", type=Path, default=DEFAULT_CASE)
    parser.add_argument("--prompt", default="请描述这些画面中正在发生的事情。")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--token-rate", type=float, default=160)
    parser.add_argument("--timeout-s", type=float, default=10)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.rounds < 1 or not 1 <= args.frames <= 8 or args.max_new_tokens < 1:
        parser.error("rounds/max-new-tokens must be positive; frames must be 1-8")
    if any(
        not math.isfinite(value) or value <= 0
        for value in (args.token_rate, args.timeout_s)
    ):
        parser.error("token-rate and timeout-s must be finite and positive")
    frames = load_frames(args)
    results, errors = [], []
    for index in range(args.rounds):
        try:
            result = await one_round(
                args.url,
                frames,
                args.prompt,
                dict(max_new_tokens=args.max_new_tokens, token_rate=args.token_rate),
                timeout_s=args.timeout_s,
            )
            results.append(result)
            if not args.json:
                print(
                    f'round {index + 1}: TTFT={result["ttft"]*1000:.1f} ms, '
                    f'total={result["total"]*1000:.1f} ms, ACKs={result["ack_count"]}'
                )
        except Exception as exc:
            errors.append(dict(round=index + 1, error=f"{type(exc).__name__}: {exc}"))

    def stats(key):
        values = sorted(result[key] * 1000 for result in results)
        if not values:
            return dict(p50=None, p95=None, max=None)
        return dict(
            p50=values[math.ceil(len(values) * 0.5) - 1],
            p95=values[math.ceil(len(values) * 0.95) - 1],
            max=values[-1],
        )

    passed = len(results) == args.rounds and not errors
    summary = dict(
        result="PASS" if passed else "FAIL",
        url=args.url,
        rounds=args.rounds,
        failures=len(errors),
        frames_per_round=args.frames,
        token_rate=args.token_rate,
        timeout_s=args.timeout_s,
        total_ms=stats("total"),
        ttft_ms=stats("ttft"),
        ready_ms=stats("ready"),
        acks_ms=stats("acks"),
        errors=errors,
        results=results,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
