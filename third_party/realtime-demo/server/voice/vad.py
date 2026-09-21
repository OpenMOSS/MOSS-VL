"""Voice activity helpers for the two capture modes.

- PTT (default): the client's button press/release delimits the turn. We only run
  an RMS activity meter to (a) reject accidental empty presses via a min-speech
  gate and (b) report level. No VAD model, no server-side endpointing latency.
- auto: a lightweight RMS endpointer (ported from board main.py) finalizes a turn
  after enough trailing silence. `silence_ms` maps to the frontend VAD slider.

audioop is stdlib on Python 3.12 (removed in 3.13 -> audioop-lts backfills it).
"""
from __future__ import annotations

from typing import Tuple

try:  # py3.13+ needs the audioop-lts backport
    import audioop  # type: ignore
except Exception:  # noqa: BLE001
    audioop = None  # type: ignore


def measure_pcm(pcm: bytes, sample_rate: int) -> Tuple[int, float]:
    """Return (rms, duration_ms) for a PCM16 mono chunk."""
    even_len = len(pcm) - (len(pcm) % 2)
    if even_len <= 0:
        return 0, 0.0
    if audioop is not None:
        try:
            rms = int(audioop.rms(pcm[:even_len], 2))
        except Exception:  # noqa: BLE001
            rms = 0
    else:
        # pure-python fallback RMS
        import array
        import math
        samples = array.array("h")
        samples.frombytes(pcm[:even_len])
        rms = int(math.sqrt(sum(s * s for s in samples) / len(samples))) if samples else 0
    duration_ms = even_len / max(1, sample_rate * 2) * 1000.0
    return rms, duration_ms


class ActivityMeter:
    """Tracks speech/silence accumulation for endpointing + gating."""

    def __init__(self, rms_threshold: int, min_speech_ms: float, silence_ms: float):
        self.rms_threshold = rms_threshold
        self.min_speech_ms = min_speech_ms
        self.silence_ms = silence_ms
        self.seen_speech = False
        self.speech_ms = 0.0
        self.silence_accum_ms = 0.0

    def reset(self) -> None:
        self.seen_speech = False
        self.speech_ms = 0.0
        self.silence_accum_ms = 0.0

    def update(self, rms: int, duration_ms: float) -> bool:
        """Feed a chunk. Returns True when an auto-finalize should fire."""
        if duration_ms <= 0:
            return False
        if rms >= self.rms_threshold:
            self.seen_speech = True
            self.speech_ms += duration_ms
            self.silence_accum_ms = 0.0
        elif self.seen_speech:
            self.silence_accum_ms += duration_ms
        return (
            self.seen_speech
            and self.speech_ms >= self.min_speech_ms
            and self.silence_accum_ms >= self.silence_ms
        )

    @property
    def has_enough_speech(self) -> bool:
        return self.seen_speech and self.speech_ms >= self.min_speech_ms
