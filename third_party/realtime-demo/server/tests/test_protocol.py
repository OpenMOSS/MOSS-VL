"""B1 test: wire-protocol encode/decode round-trips.

Run:  <repo>/.venv/bin/python -m server.tests.test_protocol
"""
from __future__ import annotations

import json
import sys
import threading

from server import protocol as p


def test_event_roundtrip() -> None:
    encoded = p.encode_event(p.RESPONSE_TEXT_DELTA, seq=42, response_id="resp_1", delta="你好")
    payload = json.loads(encoded)
    assert payload == {
        "v": p.PROTOCOL_VERSION,
        "type": p.RESPONSE_TEXT_DELTA,
        "seq": 42,
        "response_id": "resp_1",
        "delta": "你好",
    }, payload

    type_, parsed = p.parse_client_text(json.dumps({"type": "text.input", "text": "hi", "event_id": "evt_1"}))
    assert type_ == p.CLIENT_TEXT_INPUT and parsed["text"] == "hi" and parsed["event_id"] == "evt_1"

    # bare keepalive strings map to ping
    assert p.parse_client_text("ping") == (p.CLIENT_PING, {})

    for bad in ("{not json", "[1,2]", json.dumps({"no_type": 1}), json.dumps({"type": 7})):
        try:
            p.parse_client_text(bad)
        except p.ProtocolError:
            pass
        else:
            raise AssertionError(f"expected ProtocolError for {bad!r}")
    print("json events: OK")


def test_binary_roundtrip() -> None:
    pcm = b"\x01\x02\x03\x04" * 40
    tag, ts, payload = p.parse_binary(p.mic_binary(pcm))
    assert (tag, ts, payload) == (p.TAG_MIC_PCM, None, pcm)

    # tiny-but-valid JPEG prefix stands in for a real image
    jpeg = b"\xff\xd8\xff\xe0" + b"\x00" * 64 + b"\xff\xd9"

    tag, ts, payload = p.parse_binary(p.video_binary(jpeg))
    assert (tag, ts, payload) == (p.TAG_VIDEO_JPEG, None, jpeg)

    tag, ts, payload = p.parse_binary(p.video_binary(jpeg, timestamp_ms=12_345))
    assert (tag, payload) == (p.TAG_VIDEO_JPEG, jpeg) and abs(ts - 12.345) < 1e-9, ts

    # large timestamp still round-trips (u32 range)
    tag, ts, payload = p.parse_binary(p.video_binary(jpeg, timestamp_ms=3_600_000))
    assert abs(ts - 3600.0) < 1e-9

    tag, ts, payload = p.parse_binary(p.audio_binary(pcm))
    assert (tag, ts, payload) == (p.TAG_TTS_PCM, None, pcm)

    for bad in (b"", b"\x01", bytes((0x7F,)) + jpeg, bytes((p.TAG_VIDEO_JPEG,)) + b"\x00" * 16):
        try:
            p.parse_binary(bad)
        except p.ProtocolError:
            pass
        else:
            raise AssertionError(f"expected ProtocolError for {bad!r}")
    print("binary frames: OK")


def test_seq() -> None:
    seq = p.Seq()
    values = []

    def worker() -> None:
        for _ in range(500):
            values.append(seq.next())

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(values) == 2000 and len(set(values)) == 2000, "seq values must be unique"
    assert min(values) == 1 and max(values) == 2000 and seq.last == 2000
    print("seq counter: OK")


def main() -> int:
    test_event_roundtrip()
    test_binary_roundtrip()
    test_seq()
    print("\nPROTOCOL TEST OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
