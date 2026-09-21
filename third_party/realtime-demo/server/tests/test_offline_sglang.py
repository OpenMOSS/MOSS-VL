"""Offline sglang adapter: prompt/media prep, SSE deltas, failover, routing.

Run:  <repo>/.venv/bin/python -m server.tests.test_offline_sglang

Boots a STUB "sglang" server (native /generate SSE with CUMULATIVE text
frames, /health, /health_generate, /get_model_info) and drives the real
`SglangOfflinePool` against it: delta re-yield (incl. a frame far beyond
aiohttp's 64 KB line cap), board-parity sampling params, `<|image|>`/`<|video|>`
placeholder flattening + CAS blob-path resolution, replica failover on a dead
port, health→is_loaded, and the routers/chat.py online-fallback selector.
No GPU, no model, no transformers load (the chat-template owner is preset).
"""
from __future__ import annotations

# env BEFORE any server import — Settings() is env-driven and cached
import os  # noqa: E402
import tempfile  # noqa: E402

_DATA_DIR = tempfile.mkdtemp(prefix="offline_sglang_test_")
os.environ.update({
    "DATA_DIR": _DATA_DIR,             # resolve_blob_path falls back to settings
    "ASR_ENABLED": "0",
    "TTS_ENABLED": "0",
    "TTS_SPAWN": "0",
    "AUTOLOAD_VLM": "0",
})

import asyncio  # noqa: E402
import base64  # noqa: E402
import dataclasses  # noqa: E402
import hashlib  # noqa: E402
from io import BytesIO  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from typing import Any, Dict, List, Optional  # noqa: E402

import uvicorn  # noqa: E402
from fastapi import FastAPI, Request  # noqa: E402
from fastapi.responses import StreamingResponse  # noqa: E402

from server.adapters.vlm.moss_vl_sglang.adapter import (  # noqa: E402
    SglangOfflinePool, _prepare_sglang_chat, _resolve_image_payload, _sampling_params)
from server.config import Settings  # noqa: E402
from server.gpu.placement import OfflineSpec, PlacementPlan  # noqa: E402
from server.schemas import ChatRequest, GenerationParams  # noqa: E402

PORT = 19480
DEAD_PORT = 19481  # never listened on — connect-failure failover target
FAKE_MODEL = "/fake/offline/model"

# scripted CUMULATIVE texts; the last one exceeds aiohttp's ~64 KB line cap to
# prove the hand-rolled b"\n\n" frame parser is load-bearing
LONG_TAIL = "长" * 70_000
SCRIPT = ["你好", "你好，我看到", "你好，我看到一张图。", "你好，我看到一张图。" + LONG_TAIL]

captured: Dict[str, Any] = {}


class FakeTemplateOwner:
    """apply_chat_template stand-in — joins flattened contents Qwen-style."""

    def apply_chat_template(self, messages: List[Dict[str, Any]], tokenize: bool = False,
                            add_generation_prompt: bool = False) -> str:
        assert not tokenize and add_generation_prompt
        rendered = "".join(
            f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        return rendered + "<|im_start|>assistant\n"


def make_stub() -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    async def health():
        return {}

    @app.get("/health_generate")
    async def health_generate():
        return {}

    @app.get("/get_model_info")
    async def model_info():
        return {"model_path": FAKE_MODEL}

    @app.post("/generate")
    async def generate(request: Request):
        captured["body"] = await request.json()

        async def frames():
            for cum in SCRIPT:
                yield f"data: {json.dumps({'text': cum, 'meta_info': {}}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(frames(), media_type="text/event-stream")

    return app


async def start_stub() -> uvicorn.Server:
    server = uvicorn.Server(uvicorn.Config(
        make_stub(), host="127.0.0.1", port=PORT, log_level="error"))
    task = asyncio.get_running_loop().create_task(server.serve())
    while not server.started:
        if task.done():
            task.result()
        await asyncio.sleep(0.05)
    return server


def make_pool(ports: List[int]) -> SglangOfflinePool:
    settings = dataclasses.replace(
        Settings(), offline_provider="sglang", offline_model_path=FAKE_MODEL)
    plan = PlacementPlan(
        gpus=(), workers=(), tts=(), asr_device="cpu",
        offline=tuple(OfflineSpec(i, i, p) for i, p in enumerate(ports)))
    pool = SglangOfflinePool(settings, plan)
    pool._template_owner = FakeTemplateOwner()  # no transformers load in tests
    for i in range(len(ports)):
        pool.set_replica_health(i, {"ok": True})
    return pool


def put_blob(payload: bytes) -> str:
    """Write a CAS blob under the test DATA_DIR; return its sha256: handle."""
    hex_ = hashlib.sha256(payload).hexdigest()
    path = os.path.join(_DATA_DIR, "media", "blobs", "sha256", hex_[:2], hex_[2:4], hex_)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(payload)
    return f"sha256:{hex_}"


def expected_deltas() -> List[str]:
    out, prev = [], ""
    for cum in SCRIPT:
        out.append(cum[len(prev):])
        prev = cum
    return [d for d in out if d]


async def test_stream_and_params() -> None:
    pool = make_pool([PORT])
    req = ChatRequest(
        messages=[{"role": "user", "content": "描述一下"}],
        params=GenerationParams(do_sample=False, max_new_tokens=128))
    deltas = [d async for d in pool.generate_stream(req)]
    assert deltas == expected_deltas(), f"deltas mismatch: {[d[:20] for d in deltas]}"

    body = captured["body"]
    assert body["stream"] is True
    assert body["text"].endswith("<|im_start|>assistant\n")
    assert "描述一下" in body["text"]
    sp = body["sampling_params"]
    assert sp["temperature"] == 0.0            # do_sample=False → greedy
    assert sp["stop"] == ["<|im_end|>"]
    assert sp["skip_special_tokens"] is True
    assert sp["max_new_tokens"] == 128
    assert "image_data" not in body and "video_data" not in body
    print("stream + sampling params: OK")


def image_bytes(fmt: str = "JPEG") -> bytes:
    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", (16, 16), (180, 60, 30)).save(buf, format=fmt)
    return buf.getvalue()


def test_image_payloads() -> None:
    for fmt, mime in [("JPEG", "image/jpeg"), ("PNG", "image/png"),
                      ("WEBP", "image/webp"), ("GIF", "image/gif")]:
        data = image_bytes(fmt)
        encoded = base64.b64encode(data).decode("ascii")
        if fmt == "JPEG":
            assert encoded.startswith("/9j/")  # previously mistaken for a path
        expected = f"data:{mime};base64,{encoded}"
        for payload in (put_blob(data), encoded, expected):
            actual = _resolve_image_payload(payload)
            assert actual == expected
            assert base64.b64decode(actual.split(",", 1)[1], validate=True) == data
    print("image data URLs (CAS + raw base64 + existing URLs): OK")


async def test_multiturn_media_wire() -> None:
    jpeg = image_bytes()
    png = image_bytes("PNG")
    jpeg_url = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")
    png_url = "data:image/png;base64," + base64.b64encode(png).decode("ascii")
    video = put_blob(b"fake-mp4-bytes")
    messages = [
        {"role": "user", "content": [
            {"type": "image", "media": put_blob(jpeg)},
            {"type": "video", "media": video},
            {"type": "text", "text": "previous media"}]},
        {"role": "assistant", "content": "previous reply"},
        {"role": "user", "content": [
            {"type": "image", "image": base64.b64encode(png).decode("ascii")},
            {"type": "image", "image": jpeg_url},
            {"type": "text", "text": "compare with previous media"}]},
    ]
    req = ChatRequest(messages=messages)
    before = req.model_dump()
    assert [d async for d in make_pool([PORT]).generate_stream(req)] == expected_deltas()
    body = captured["body"]
    assert body["image_data"] == [jpeg_url, png_url, jpeg_url]
    assert len(body["video_data"]) == 1 and os.path.exists(body["video_data"][0])
    assert body["text"].count("<|image|>") == 3
    assert body["text"].count("<|video|>") == 1
    assert "previous reply" in body["text"] and "compare with previous media" in body["text"]
    assert req.model_dump() == before
    print("multi-turn media preserved in actual HTTP payload: OK")


async def test_media_prep() -> None:
    jpeg = image_bytes()
    img = put_blob(jpeg)
    vid = put_blob(b"fake-mp4-bytes")
    req = ChatRequest(
        messages=[{"role": "user", "content": [
            {"type": "image", "media": img},
            {"type": "video", "media": vid},
            {"type": "text", "text": "这是什么？"},
        ]}],
        images=["data:image/png;base64,AAAA"])  # legacy top-level → same message
    messages, image_data, video_data = _prepare_sglang_chat(req)
    content = messages[-1]["content"]
    assert content == "<|image|><|image|><|video|>这是什么？", content
    assert len(image_data) == 2
    assert image_data[0] == "data:image/png;base64,AAAA"  # data-URL passes through
    # Images use explicit data URLs; videos keep their existing blob paths.
    assert image_data[1] == "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")
    assert len(video_data) == 1 and os.path.exists(video_data[0])

    # user-typed placeholder suppresses ONE auto-insertion (board skip logic)
    req = ChatRequest(messages=[{"role": "user", "content": [
        {"type": "image", "media": img},
        {"type": "text", "text": "看这个 <|image|> 旁边的字"},
    ]}])
    messages, image_data, _ = _prepare_sglang_chat(req)
    assert messages[-1]["content"].count("<|image|>") == 1
    assert len(image_data) == 1

    # videos must be CAS handles — anything else is rejected
    try:
        _prepare_sglang_chat(ChatRequest(
            messages=[{"role": "user", "content": [
                {"type": "video", "media": "/etc/passwd"}]}]))
    except ValueError:
        pass
    else:
        raise AssertionError("non-handle video payload must raise")
    # unknown handle (no blob on disk) is rejected too
    try:
        _prepare_sglang_chat(ChatRequest(
            messages=[{"role": "user", "content": [
                {"type": "image", "media": "sha256:" + "0" * 64}]}]))
    except ValueError:
        pass
    else:
        raise AssertionError("unknown image handle must raise")
    print("media prep: OK")


async def test_failover_and_down() -> None:
    # replica 0 (stale-healthy, dead port) → connect failure → replica 1 serves
    pool = make_pool([DEAD_PORT, PORT])
    req = ChatRequest(messages=[{"role": "user", "content": "hi"}])
    deltas = [d async for d in pool.generate_stream(req)]
    assert deltas == expected_deltas()

    # all replicas down → is_loaded False, generate_stream raises
    pool.set_replica_health(0, None)
    pool.set_replica_health(1, None)
    assert not pool.is_loaded()
    st = pool.status()
    assert st["loaded"] is False and st["capacity"] == 2
    assert all(r["state"] == "down" for r in st["replicas"])
    try:
        async for _ in pool.generate_stream(req):
            pass
    except RuntimeError:
        pass
    else:
        raise AssertionError("all-down pool must raise")
    print("failover + down: OK")


async def test_chat_routing() -> None:
    from server.deps import Runtime
    from server.routers.chat import _chat_vlm
    from server.tests.fakes import FakeOfflineVlm, FakeVlmAdapter

    rt = Runtime.__new__(Runtime)  # partial Runtime (repo test convention)
    rt.vlm = FakeVlmAdapter()
    rt.vlm_offline = FakeOfflineVlm(loaded=True)
    assert _chat_vlm(rt) is rt.vlm_offline
    rt.vlm_offline.loaded = False
    assert _chat_vlm(rt) is rt.vlm            # plane down → online fallback
    rt2 = Runtime.__new__(Runtime)
    rt2.vlm = FakeVlmAdapter()
    assert _chat_vlm(rt2) is rt2.vlm          # no plane at all (1-GPU box)
    print("chat routing: OK")


def test_sampling_unit() -> None:
    sp = _sampling_params(GenerationParams(
        temperature=0.9, do_sample=True, top_k=5, top_p=0.5,
        repetition_penalty=1.1, max_new_tokens=64))
    assert sp == {"max_new_tokens": 64, "temperature": 0.9, "top_p": 0.5,
                  "top_k": 5, "repetition_penalty": 1.1,
                  "stop": ["<|im_end|>"], "skip_special_tokens": True}
    # schema default is now repetition_penalty=None (realtime falls back to
    # GEN_REPETITION_PENALTY) — the sglang mapper must stay None-safe
    assert _sampling_params(GenerationParams())["repetition_penalty"] == 1.0
    print("sampling unit: OK")


async def amain() -> int:
    stub = await start_stub()
    try:
        await test_stream_and_params()
        await test_media_prep()
        test_image_payloads()
        await test_multiturn_media_wire()
        await test_failover_and_down()
        await test_chat_routing()
        test_sampling_unit()
    finally:
        stub.should_exit = True
        await asyncio.sleep(0.1)
    print("\nOFFLINE SGLANG TEST OK")
    return 0


def main() -> int:
    return asyncio.run(amain())


if __name__ == "__main__":
    sys.exit(main())
