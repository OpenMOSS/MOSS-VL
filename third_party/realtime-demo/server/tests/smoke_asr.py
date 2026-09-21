"""M1.1 smoke: FunASR SenseVoice loads on GPU, warms up, and decodes a turn.

Run:  ASR_DEVICE=cuda:0 <repo>/.venv/bin/python -m server.tests.smoke_asr

Validates the ASR adapter end to end (GPU fp16 load + warmup + turn-buffered
finalize). Decoding synthetic audio may yield empty/garbage text — the check is
that load/warmup/finalize complete without error.
"""
from __future__ import annotations

import sys
import time

from server.config import get_settings
from server.adapters.asr.funasr_sensevoice.adapter import FunasrSenseVoiceAdapter


def main() -> int:
    settings = get_settings()
    adapter = FunasrSenseVoiceAdapter(settings)
    t0 = time.monotonic()
    adapter.start()
    print(f"start() -> ready={adapter.ready} in {time.monotonic() - t0:.1f}s :: {adapter.status().get('message')}")
    if not adapter.ready:
        print("ASR not ready — aborting smoke")
        return 1

    # 1.2 s of low-level noise so the turn passes the min-pcm gate
    import math
    sr = settings.asr_sample_rate
    n = int(sr * 1.2)
    pcm = bytearray()
    for i in range(n):
        v = int(1500 * math.sin(2 * math.pi * 220 * i / sr))
        pcm += int(v).to_bytes(2, "little", signed=True)

    stream = adapter.open_stream()
    # feed as 160 ms chunks like the real client
    chunk = int(sr * 0.160) * 2
    for off in range(0, len(pcm), chunk):
        stream.send_pcm(bytes(pcm[off:off + chunk]))
    t1 = time.monotonic()
    text = stream.finalize()
    stream.close()
    print(f"finalize() -> {time.monotonic() - t1:.2f}s, text={text!r}")
    print("\nASR SMOKE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
