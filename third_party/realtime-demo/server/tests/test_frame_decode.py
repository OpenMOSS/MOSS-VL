"""Unit tests: corrupt JPEG frames must not kill a realtime session.

Run:  <repo>/.venv/bin/python -m server.tests.test_frame_decode

Regression: the frame-bytes protocol moved JPEG decode inside
RealtimeSession.put_frame/put_prompt_frame. A truncated JPEG passes the
gateway's cheap SOI check, so the decode failure lands here — it must drop
the frame (put_frame) or degrade to a text-only turn (put_prompt_frame),
never raise (a raise tears down the worker io WS and quarantines the replica).
"""
from __future__ import annotations

import queue
import sys
import threading
import time
from io import BytesIO

from server.realtime.session import (
    RealTimeFrameQueue,
    RealTimeOutputQueue,
    RealtimeSession,
)

# valid SOI marker, then garbage — passes looks_like_jpeg, fails PIL decode
BAD_JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 64


def good_jpeg() -> bytes:
    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", (32, 32), (200, 40, 40)).save(buf, format="JPEG")
    return buf.getvalue()


def make_session() -> RealtimeSession:
    now = time.time()
    return RealtimeSession(
        session_id="test-frames",
        gpu_id=-1,
        frame_queue=RealTimeFrameQueue(maxsize=8),
        prompt_queue=queue.Queue(),
        output_queue=RealTimeOutputQueue(created_at=now),
        stop_event=threading.Event(),
        created_at=now,
    )


def test_bad_frame_is_dropped() -> None:
    session = make_session()
    payload = session.put_frame(BAD_JPEG, 1.0)
    assert payload.get("bad_frame") is True
    assert session.frames_dropped == 1 and session.frames_received == 1
    assert session.frame_queue.qsize() == 0, "a bad frame must never reach the model"

    # the session survives and still accepts valid frames
    payload = session.put_frame(good_jpeg(), 2.0)
    assert not payload.get("bad_frame")
    assert session.frame_queue.qsize() == 1 and session.frames_dropped == 1
    print("bad frame dropped, session survives: OK")


def test_bad_turn_frame_degrades_to_prompt() -> None:
    session = make_session()
    session.put_prompt_frame("描述画面", BAD_JPEG, 1.0)
    assert session.prompts_received == 1 and session.frames_dropped == 1
    # the frame dropped, but the user's turn must still be delivered — as a
    # string on new_prompts (the model reads prompts there, not the frame queue)
    assert session.frame_queue.qsize() == 0, "a bad frame must never reach the model"
    assert session.prompt_queue.get_nowait() == "描述画面"
    print("bad turn frame degrades to text-only prompt: OK")


def test_decoded_bytes_reach_model_as_pil() -> None:
    session = make_session()
    session.put_frame(good_jpeg(), 1.0)
    # the model drains new_video_frames as (PIL.Image, float ts) tuples
    image, ts = session.frame_queue.get_nowait()
    assert getattr(image, "size", None) == (32, 32) and ts == 1.0
    assert session.last_frame_size == (32, 32)
    print("valid bytes decode to PIL at the model boundary: OK")


def main() -> int:
    test_bad_frame_is_dropped()
    test_bad_turn_frame_degrades_to_prompt()
    test_decoded_bytes_reach_model_as_pil()
    print("\nFRAME DECODE TEST OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
