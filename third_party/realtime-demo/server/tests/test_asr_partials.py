"""_SenseVoiceStream partial worker: periodic re-decode → on_partial callback.

Run:  <repo>/.venv/bin/python -m server.tests.test_asr_partials

Uses a stub engine — no funasr/GPU. Verifies: partials fire while audio grows,
idle buffers are not re-decoded, finalize stops the worker and returns the
full-buffer decode, and interval=0 never spawns a worker.
"""
from __future__ import annotations

import sys
import time

from server.adapters.asr.funasr_sensevoice.adapter import _SenseVoiceStream


class StubEngine:
    min_pcm_bytes = 4

    def __init__(self, partial_interval_ms: int = 30):
        self.partial_interval_ms = partial_interval_ms
        self.decodes = 0

    def transcribe_pcm(self, pcm: bytes, allow_short: bool = False) -> str:
        self.decodes += 1
        return f"hyp-{len(pcm)}"


def wait_for(cond, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.01)


def test_partial_window_and_ceiling() -> None:
    # partials re-decode only the RECENT window (bounded), not the whole turn
    win = StubEngine(partial_interval_ms=20)
    win.partial_window_bytes = 16  # 8 bytes = one send below
    win.buffer_max_bytes = 0       # memory unbounded here — isolate the window
    got: list = []
    stream = _SenseVoiceStream(win, on_partial=got.append)
    for _ in range(3):
        stream.send_pcm(b"\x01\x00" * 4)  # 8 bytes each → total 24
        time.sleep(0.03)
    wait_for(lambda: "hyp-16" in got)
    assert "hyp-24" not in got, ("partial must cap at the 16-byte window", got)
    # the FINAL still decodes the whole retained turn (window bounds partials only)
    assert stream.finalize() == "hyp-24", "final decodes the full buffer"
    stream.close()

    # memory ceiling drops only the OLDEST audio so long turns stay bounded
    cap = StubEngine(partial_interval_ms=10_000)  # no partial interference
    cap.partial_window_bytes = 0
    cap.buffer_max_bytes = 16
    s = _SenseVoiceStream(cap, on_partial=None)
    for _ in range(4):
        s.send_pcm(b"\x01\x00" * 4)  # 8 bytes each → 32, ceiling 16
    assert s.pcm_len() <= 16, ("ceiling must bound the retained buffer", s.pcm_len())
    assert s.finalize() == "hyp-16"
    s.close()
    print("partial window + memory ceiling: OK")


def main() -> int:
    # partials fire and track the growing buffer
    engine = StubEngine()
    got: list = []
    stream = _SenseVoiceStream(engine, on_partial=got.append)
    stream.send_pcm(b"\x01\x00" * 8)
    wait_for(lambda: "hyp-16" in got)
    stream.send_pcm(b"\x01\x00" * 8)
    wait_for(lambda: got[-1] == "hyp-32")

    # no new audio → no further decodes (the buffer-growth gate)
    decoded_at = engine.decodes
    time.sleep(0.15)
    assert engine.decodes == decoded_at, "idle buffer must not be re-decoded"

    # finalize stops the worker and decodes the whole buffer once more
    assert stream.finalize() == "hyp-32"
    n_partials = len(got)
    time.sleep(0.1)
    assert len(got) == n_partials, "no partials after finalize"
    stream.close()
    print("partials + idle gate + finalize: OK")

    # interval=0 (or no callback) → turn-level behavior, no worker thread
    for kwargs in ({"on_partial": None}, {"on_partial": got.append}):
        s = _SenseVoiceStream(StubEngine(partial_interval_ms=0), **kwargs)
        s.send_pcm(b"\x01\x00" * 8)
        assert s._partial_thread is None
        s.close()
    print("disabled partials spawn no worker: OK")

    test_partial_window_and_ceiling()

    print("\nASR PARTIALS TEST OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
