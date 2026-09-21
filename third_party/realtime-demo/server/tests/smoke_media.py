"""Media + history HTTP smoke over a live uvicorn (fake engines, real persistence).

Run:  <repo>/.venv/bin/python -m server.tests.smoke_media

Covers: multipart upload → CAS descriptor · dedup (re-POST → same hash) ·
magic-byte reject (415) · size cap (413) · GET blob with ETag/immutable ·
If-None-Match → 304 · Range → 206 · thumbnail · chat SSE with conversation_id →
history recorded · GET/DELETE /api/history.
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from io import BytesIO
from typing import Optional, Tuple

import uvicorn
from PIL import Image

from server.config import Settings
from server.deps import Runtime
from server.persistence import HistoryRecorder, IndexStore, MediaStore, set_media_store
from server.session.manager import SessionManager
from server.tests.fakes import FakeAsrAdapter, FakeTtsAdapter, FakeVlmAdapter

UPLOAD_CAP = 64 * 1024


def make_runtime(data_dir: str) -> Runtime:
    settings = dataclasses.replace(Settings(), data_dir=data_dir, history_db_path="",
                                   upload_max_bytes=UPLOAD_CAP)
    rt = Runtime.__new__(Runtime)
    rt.settings = settings
    rt.asr = FakeAsrAdapter()
    rt.tts = FakeTtsAdapter()
    rt.vlm = FakeVlmAdapter(token_interval=0.005)
    rt.index = IndexStore(settings)
    rt.media = MediaStore(settings, rt.index)
    rt.history = HistoryRecorder(settings, rt.index)
    rt.session_manager = SessionManager(settings, history=rt.history)
    rt._tts_sessions = {}
    rt._lock = threading.Lock()
    return rt


def http(method: str, url: str, body: Optional[bytes] = None,
         headers: Optional[dict] = None) -> Tuple[int, dict, bytes]:
    req = urllib.request.Request(url, data=body, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, {k.lower(): v for k, v in resp.headers.items()}, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, {k.lower(): v for k, v in exc.headers.items()}, exc.read()


def multipart(data: bytes, filename: str, mime: str) -> Tuple[bytes, str]:
    boundary = "----mossmediasmoke"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: {mime}\r\n\r\n"
    ).encode() + data + f"\r\n--{boundary}--\r\n".encode()
    return body, f"multipart/form-data; boundary={boundary}"


def jpeg_bytes(size=(300, 180)) -> bytes:
    buf = BytesIO()
    Image.new("RGB", size, (20, 90, 210)).save(buf, format="JPEG")
    return buf.getvalue()


async def start_server(rt: Runtime):
    from server import app as app_module

    app_module.build_runtime = lambda plan=None: rt
    config = uvicorn.Config(app_module.app, host="127.0.0.1", port=0,
                            log_level="warning", lifespan="on")
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.02)
        assert not task.done(), "uvicorn failed to start"
    port = server.servers[0].sockets[0].getsockname()[1]
    return server, task, f"http://127.0.0.1:{port}"


async def main_async() -> None:
    with tempfile.TemporaryDirectory(prefix="media_smoke_") as data_dir:
        rt = make_runtime(data_dir)
        server, task, base = await start_server(rt)
        try:
            # ---- upload ----
            img = jpeg_bytes()
            body, ctype = multipart(img, "photo.jpg", "image/jpeg")
            code, _, raw = await asyncio.to_thread(
                http, "POST", f"{base}/api/media", body, {"Content-Type": ctype})
            assert code == 200, (code, raw)
            desc = json.loads(raw)
            assert desc["hash"].startswith("sha256:") and desc["kind"] == "image"
            assert desc["thumb_url"], desc
            hex_ = desc["hash"].split(":", 1)[1]
            print("upload: OK", desc["hash"][:20])

            # dedup: identical bytes → identical identity
            code, _, raw2 = await asyncio.to_thread(
                http, "POST", f"{base}/api/media", body, {"Content-Type": ctype})
            assert code == 200 and json.loads(raw2)["hash"] == desc["hash"]
            print("dedup: OK")

            # reject: text with a .jpg name (magic bytes win) → 415
            bad, ctype2 = multipart(b"plain text pretending " * 20, "fake.jpg", "image/jpeg")
            code, _, raw = await asyncio.to_thread(
                http, "POST", f"{base}/api/media", bad, {"Content-Type": ctype2})
            assert code == 415, (code, raw)
            print("magic-byte reject 415: OK")

            # reject: over the cap → 413 (valid JPEG head + padding past the cap)
            big, ctype3 = multipart(jpeg_bytes() + b"\x00" * UPLOAD_CAP, "big.jpg", "image/jpeg")
            code, _, raw = await asyncio.to_thread(
                http, "POST", f"{base}/api/media", big, {"Content-Type": ctype3})
            assert code == 413, (code, raw)
            print("size cap 413: OK")

            # ---- serving ----
            code, headers, blob = await asyncio.to_thread(http, "GET", f"{base}/api/media/{hex_}")
            assert code == 200 and headers["etag"] == f'"sha256:{hex_}"'
            assert "immutable" in headers["cache-control"]
            assert headers["x-content-type-options"] == "nosniff"
            assert headers["content-type"] == "image/jpeg"
            stored = blob  # re-encoded bytes (EXIF-strip) — decodable JPEG
            Image.open(BytesIO(stored)).verify()
            print("GET blob + headers: OK")

            code, _, _ = await asyncio.to_thread(
                http, "GET", f"{base}/api/media/{hex_}", None,
                {"If-None-Match": f'"sha256:{hex_}"'})
            assert code == 304, code
            print("If-None-Match 304: OK")

            code, headers, part = await asyncio.to_thread(
                http, "GET", f"{base}/api/media/{hex_}", None, {"Range": "bytes=0-99"})
            assert code == 206 and len(part) == 100, (code, len(part))
            assert headers["content-range"] == f"bytes 0-99/{len(stored)}"
            assert part == stored[:100]
            code, _, tail = await asyncio.to_thread(
                http, "GET", f"{base}/api/media/{hex_}", None, {"Range": "bytes=-50"})
            assert code == 206 and tail == stored[-50:]
            code, _, _ = await asyncio.to_thread(
                http, "GET", f"{base}/api/media/{hex_}", None,
                {"Range": f"bytes={len(stored) + 10}-"})
            assert code == 416
            print("Range 206/416: OK")

            code, headers, thumb = await asyncio.to_thread(
                http, "GET", f"{base}/api/media/{hex_}/thumb")
            assert code == 200 and headers["content-type"] == "image/jpeg"
            Image.open(BytesIO(thumb)).verify()
            print("thumbnail: OK")

            code, _, _ = await asyncio.to_thread(http, "GET", f"{base}/api/media/{'0' * 64}")
            assert code == 404
            code, _, _ = await asyncio.to_thread(http, "GET", f"{base}/api/media/nothex")
            assert code == 400
            print("404/400 paths: OK")

            # ---- chat with conversation_id → history over HTTP ----
            cid = "smoke_thread_001"
            chat = json.dumps({
                "messages": [{"role": "user", "content": "看看这张图片里有什么"}],
                "images": [desc["hash"]],
                "conversation_id": cid,
            }).encode()
            code, _, raw = await asyncio.to_thread(
                http, "POST", f"{base}/api/chat/stream", chat,
                {"Content-Type": "application/json"})
            assert code == 200 and b"generation_end" in raw, (code, raw[:200])
            await asyncio.sleep(0.3)  # writer thread drain

            code, _, raw = await asyncio.to_thread(http, "GET", f"{base}/api/history")
            convs = json.loads(raw)["conversations"]
            assert any(c["conversation_id"] == cid for c in convs), convs
            code, _, raw = await asyncio.to_thread(http, "GET", f"{base}/api/history/{cid}")
            tr = json.loads(raw)
            assert tr["title"] == "看看这张图片里有什么"
            assert [t["role"] for t in tr["turns"]] == ["user", "assistant"]
            assert tr["turns"][0]["media"][0]["hash"] == desc["hash"]
            assert tr["turns"][1]["metrics"]["latency_ms"] > 0
            print("chat → history recording: OK")

            code, _, raw = await asyncio.to_thread(
                http, "GET", f"{base}/api/history?q=%E8%BF%99%E5%BC%A0%E5%9B%BE")  # 这张图
            assert any(c["conversation_id"] == cid
                       for c in json.loads(raw)["conversations"]), raw
            print("history FTS search: OK")

            # ---- multi-turn follow-up: parts-form content, same conversation ----
            chat2 = json.dumps({
                "messages": [
                    {"role": "user", "content": [
                        {"type": "image", "media": desc["hash"]},
                        {"type": "text", "text": "看看这张图片里有什么"}]},
                    {"role": "assistant", "content": "一张蓝色的图片"},
                    {"role": "user", "content": [
                        {"type": "image", "media": desc["hash"]},
                        {"type": "text", "text": "它的背景是什么颜色？"}]},
                ],
                "conversation_id": cid,
            }).encode()
            code, _, raw = await asyncio.to_thread(
                http, "POST", f"{base}/api/chat/stream", chat2,
                {"Content-Type": "application/json"})
            assert code == 200 and b"generation_end" in raw, (code, raw[:200])
            await asyncio.sleep(0.3)
            code, _, raw = await asyncio.to_thread(http, "GET", f"{base}/api/history/{cid}")
            tr2 = json.loads(raw)
            # only the LAST user message is recorded as the new turn (earlier
            # messages are resent context) + the new assistant reply
            assert [t["role"] for t in tr2["turns"]] == ["user", "assistant", "user", "assistant"], \
                [t["role"] for t in tr2["turns"]]
            assert tr2["turns"][2]["text"] == "它的背景是什么颜色？"
            assert tr2["turns"][2]["media"][0]["hash"] == desc["hash"]
            print("multi-turn follow-up recorded: OK")

            # bad conversation_id → 400
            bad_chat = json.dumps({
                "messages": [{"role": "user", "content": "hi"}],
                "conversation_id": "../evil",
            }).encode()
            code, _, _ = await asyncio.to_thread(
                http, "POST", f"{base}/api/chat/stream", bad_chat,
                {"Content-Type": "application/json"})
            assert code == 400, code
            print("bad conversation_id 400: OK")

            code, _, raw = await asyncio.to_thread(http, "DELETE", f"{base}/api/history/{cid}")
            assert code == 200, (code, raw)
            code, _, _ = await asyncio.to_thread(http, "GET", f"{base}/api/history/{cid}")
            assert code == 404
            print("history delete: OK")
        finally:
            server.should_exit = True
            await task
            set_media_store(None)

    print("\nMEDIA SMOKE OK")


def main() -> int:
    asyncio.run(main_async())
    return 0


if __name__ == "__main__":
    sys.exit(main())
