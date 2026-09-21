"""Capture one verbatim realtime session through the gateway → JSONL of every event.

P7 raw material: the protocol doc's per-event JSON examples should come from a
REAL wire capture, not hand-writing. This script drives one session through the
REST+WSS gateway plane (create → attach → configure → frames → prompt → final
→ done) and dumps every message in both directions, in order, as JSONL:

    {"dir": "down", "event": {...}}        # server → client (verbatim)
    {"dir": "up",   "event": {...}}        # client → server (metadata only)
    {"dir": "up",   "event": "<binary>", "bytes": N}   # binary frame marker

Usage (run on the GPU node, gateway local):

    .venv/bin/python tools/capture_realtime_events.py \
        --url http://127.0.0.1:8100 --frames data/stress_frames \
        --n-frames 6 --prompt "描述一下画面里发生了什么" \
        --out reports/capture_session.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
import urllib.request
from pathlib import Path

import websockets


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8100")
    ap.add_argument("--frames", required=True, help="dir of .jpg files")
    ap.add_argument("--n-frames", type=int, default=6)
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--prompt", default="描述一下画面里发生了什么")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    frames = sorted(Path(args.frames).glob("*.jpg"))[: args.n_frames]
    assert frames, f"no .jpg in {args.frames}"
    out = open(args.out, "w", encoding="utf-8")

    def rec(direction: str, event) -> None:
        out.write(json.dumps({"dir": direction, "event": event},
                             ensure_ascii=False) + "\n")
        out.flush()

    # REST create
    req = urllib.request.Request(f"{args.url}/v1/realtime/sessions", data=b"{}",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        created = json.loads(resp.read())
    rec("rest", {"POST /v1/realtime/sessions ->": created})
    ws_url = args.url.replace("http://", "ws://") + created["ws_url"]

    async with websockets.connect(
            f"{ws_url}?ws_token={created['ws_token']}", max_size=64 * 1024 * 1024) as ws:
        first = json.loads(await ws.recv())  # session.created
        rec("down", first)

        async def send_meta(ev: dict) -> None:
            rec("up", ev)
            await ws.send(json.dumps(ev, ensure_ascii=False))

        await send_meta({
            "type": "session.configure",
            "prompt": "请持续观察画面并描述变化。",
            "system_prompt": None,
            "max_new_tokens": 4096,
            "temperature": 0.0,
            "top_p": 1.0,
            "input_queue_capacity": 4,
        })

        # pump: react to events; send frames on ready; prompt after frames
        seq = 0
        frame_idx = 0
        ts = 0.0
        prompted = False
        finaled = False
        done = asyncio.Event()
        pending_seq = None

        async def next_input() -> None:
            nonlocal seq, frame_idx, ts, prompted, finaled, pending_seq
            if frame_idx < len(frames):
                await send_meta({
                    "type": "input.frame", "seq_no": seq, "timestamp": round(ts, 1),
                    "final": False, "mime_type": "image/jpeg"})
                pending_seq = seq
                seq += 1
                ts += 1.0 / args.fps
            elif not prompted:
                await send_meta({"type": "input.prompt", "seq_no": seq,
                                 "prompt": args.prompt, "final": False})
                prompted = True
                seq += 1
            elif not finaled:
                # one more frame with final=true → normal end path
                await send_meta({
                    "type": "input.frame", "seq_no": seq, "timestamp": round(ts, 1),
                    "final": True, "mime_type": "image/jpeg"})
                pending_seq = seq
                finaled = True
                seq += 1
            await asyncio.sleep(1.0 / args.fps)

        try:
            async with asyncio.timeout(120):
                while not done.is_set():
                    raw = await ws.recv()
                    if isinstance(raw, (bytes, bytearray)):
                        rec("down", "<unexpected-binary>")
                        continue
                    ev = json.loads(raw)
                    rec("down", ev)
                    etype = ev.get("type")
                    if etype == "session.ready":
                        await next_input()  # first input only after ready
                    elif etype == "input.frame.ready" and pending_seq is not None:
                        # the final extra frame reuses the last image
                        payload = frames[min(frame_idx, len(frames) - 1)].read_bytes()
                        rec("up", {"event": "<binary-frame>", "bytes": len(payload)})
                        await ws.send(payload)
                        frame_idx += 1
                    elif etype in ("input.frame.accepted", "input.prompt.accepted"):
                        if not (finaled and prompted):
                            await next_input()
                    elif etype in ("session.done", "error"):
                        done.set()
        except (TimeoutError, asyncio.TimeoutError):
            rec("note", "capture timed out — closing")
    rec("rest", {"GET /v1/realtime/sessions/{id} ->": "see gateway"})
    print(f"capture written: {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
