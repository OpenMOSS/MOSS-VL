#!/usr/bin/env python3
"""Replay images and questions without treating per-response done as session end."""
import argparse
import asyncio
import json
import math
import os
from contextlib import suppress
from io import BytesIO
from pathlib import Path

from PIL import Image
from websockets.asyncio.client import connect


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://127.0.0.1:18610/v1/video/realtime")
    parser.add_argument("--frame", type=Path, action="append", required=True)
    parser.add_argument("--prompt", action="append", default=None)
    parser.add_argument("--timeout", type=float, default=60)
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("timeout must be finite and positive")
    headers = {}
    key = os.environ.get("VL_API_V2_API_KEY")
    if key:
        headers["Authorization"] = "Bearer " + key
    async with connect(
        args.url,
        proxy=None,
        additional_headers=headers,
        open_timeout=10,
        close_timeout=5,
    ) as ws:
        created = json.loads(await asyncio.wait_for(ws.recv(), 10))
        print(json.dumps(created, ensure_ascii=False), flush=True)
        if "response.done.per_response" not in created.get("capabilities", []):
            raise RuntimeError(
                "the selected endpoint does not advertise per-response v2"
            )
        arrivals = asyncio.Queue()

        async def receive():
            try:
                async for raw in ws:
                    event = json.loads(raw)
                    print(json.dumps(event, ensure_ascii=False), flush=True)
                    await arrivals.put(event)
            except Exception as exc:
                await arrivals.put({"type": "error", "message": str(exc)})
            finally:
                await arrivals.put({"type": "transport.closed"})

        reader = asyncio.create_task(receive())

        async def wait_for(kind, seq=None, turn=None):
            async with asyncio.timeout(args.timeout):
                while True:
                    item = await arrivals.get()
                    if item["type"] == "error":
                        raise RuntimeError(item.get("message"))
                    if (
                        item["type"] == kind
                        and (seq is None or item.get("seq_no") == seq)
                        and (turn is None or item.get("turn_id") == turn)
                    ):
                        return item
                    if item["type"] in ("session.done", "transport.closed"):
                        raise RuntimeError(
                            f"session ended while waiting for {kind}: {item}"
                        )

        try:
            await ws.send(
                json.dumps(
                    {
                        "type": "session.configure",
                        "prompt": "",
                        "max_new_tokens": 512,
                        "include_usage": True,
                    }
                )
            )
            await wait_for("session.ready")
            seq = 0
            for path in args.frame:
                buffer = BytesIO()
                with Image.open(path) as image:
                    image.convert("RGB").save(buffer, format="JPEG", quality=90)
                await ws.send(
                    json.dumps(
                        {
                            "type": "input.frame",
                            "seq_no": seq,
                            "timestamp": float(seq),
                            "mime_type": "image/jpeg",
                        }
                    )
                )
                await wait_for("input.frame.ready", seq)
                await ws.send(buffer.getvalue())
                await wait_for("input.frame.accepted", seq)
                seq += 1
            prompts = args.prompt or ["请描述画面中正在发生的事情。"]
            for index, prompt in enumerate(prompts):
                final = index == len(prompts) - 1
                await ws.send(
                    json.dumps(
                        {
                            "type": "input.prompt",
                            "seq_no": seq,
                            "prompt": prompt,
                            "final": final,
                        },
                        ensure_ascii=False,
                    )
                )
                await wait_for("input.prompt.accepted", seq)
                processed = await wait_for("input.prompt.processed", seq)
                # Non-final segments keep the same backend request alive.
                await wait_for(
                    "session.done" if final else "response.done",
                    turn=None if final else processed["turn_id"],
                )
                seq += 1
        finally:
            with suppress(Exception):
                await ws.send('{"type":"session.abort"}')
            reader.cancel()
            await asyncio.gather(reader, return_exceptions=True)


if __name__ == "__main__":
    asyncio.run(main())
