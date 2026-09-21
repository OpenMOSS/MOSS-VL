"""M1.2 smoke: load streaming MOSS-VL and drive a real realtime turn.

Run:  MODEL_PATH=<realtime-sft-ckpt> GPU_ID=0 \
      <repo>/.venv/bin/python -m server.tests.smoke_realtime

Loads the streaming checkpoint via HfMossVlAdapter, starts a realtime session,
streams synthetic frames, then sends a prompt+frame user turn and polls the output
queue. Success = the model loop runs and emits output events (text or <|silence|>);
any non-silence text is a bonus that proves generation end-to-end.
"""
from __future__ import annotations

import os
import sys
import time

from PIL import Image, ImageDraw

from server.adapters.vlm.moss_vl_hf.adapter import HfMossVlAdapter
from server.config import get_settings


_REAL_IMAGE = os.environ.get("IMAGE_PATH")


def _frame(color, label: str) -> Image.Image:
    if _REAL_IMAGE:
        return Image.open(_REAL_IMAGE).convert("RGB")
    img = Image.new("RGB", (448, 336), color)
    d = ImageDraw.Draw(img)
    d.rectangle([120, 90, 328, 246], fill=color, outline=(255, 255, 255), width=6)
    d.text((140, 150), label, fill=(255, 255, 255))
    return img


def main() -> int:
    model_path = os.environ.get("MODEL_PATH")
    gpu_id = int(os.environ.get("GPU_ID", "0"))
    if not model_path:
        print("set MODEL_PATH")
        return 2

    settings = get_settings()
    adapter = HfMossVlAdapter(settings)

    t0 = time.monotonic()
    adapter.load(model_path, gpu_id, "online_streaming")
    print(f"[load] {time.monotonic() - t0:.1f}s  status={adapter.status()}")

    scenario = os.environ.get("SCENARIO", "prompt_frame")  # prompt_frame | frames_only | prompt_only
    session = adapter.start_realtime_session(
        prompt="",
        max_new_tokens=256,
        do_sample=False,
    )
    print(f"[session] started {session.session_id} scenario={scenario}")

    if scenario == "prompt_only":
        # pure text turn, no frames at all
        session.put_prompt("你好，请用一句话介绍你自己。")
    elif scenario == "frames_only":
        for i in range(5):
            session.put_frame(_frame((200, 30, 30), "RED"), timestamp=float(i))
            time.sleep(0.35)
    else:  # prompt_frame
        for i in range(3):
            session.put_frame(_frame((200, 30, 30), "RED"), timestamp=float(i))
            time.sleep(0.35)
        session.put_prompt_frame("画面主要是什么颜色？请简短回答。", _frame((200, 30, 30), "RED"), timestamp=3.0)
    print("[turn] inputs sent; polling output...")

    chunks = []
    non_silence = []
    first_out = None
    first_non_silence = None
    deadline = time.monotonic() + 40
    turn_started = time.monotonic()
    while time.monotonic() < deadline:
        batch = session.poll_output(0.5, 128)
        for ev in batch.chunk_events:
            text = ev["text"]
            chunks.append(text)
            if first_out is None:
                first_out = time.monotonic() - turn_started
            if text not in ("<|silence|>", "<|...|>", "<|round_start|>"):
                non_silence.append(text)
                if first_non_silence is None:
                    first_non_silence = time.monotonic() - turn_started
        if len("".join(non_silence)) >= 8:
            break
        if not batch.active:
            print("[warn] session became inactive")
            break

    st = session.status()
    print(f"[result] frames_recv={st['frames_received']} consumed={st['frames_consumed']} "
          f"outputs={st['outputs_emitted']} silence={st['silence_outputs']} non_silence={st['non_silence_outputs']}")
    print(f"[timing] first_output={first_out}  first_non_silence={first_non_silence}")
    print(f"[text] {''.join(non_silence)[:400]!r}")
    session.stop()

    ok = st["outputs_emitted"] > 0
    print("\nREALTIME SMOKE", "OK" if ok else "FAILED (no outputs)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
