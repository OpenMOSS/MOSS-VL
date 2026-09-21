"""Persistence tests: recorder journal+index, media CAS, delete, rebuild.

Run:  <repo>/.venv/bin/python -m server.tests.test_persistence

No GPU, no network — fake engines, temp DATA_DIRs.
"""
from __future__ import annotations

import asyncio
import dataclasses
import glob
import json
import os
import sys
import tempfile
import time
from io import BytesIO

from PIL import Image

from server.config import Settings
from server.persistence import HistoryRecorder, IndexStore, MediaStore, set_media_store
from server.persistence.media import MediaError, normalize_hash
from server.persistence.recorder import iter_journal_files, replay_file
from server.schemas import SessionConfig
from server.session.manager import SessionManager
from server.session.orchestrator import EngineSet
from server.tests.fakes import FakeAsrAdapter, FakeTtsEngine, FakeVlmSession
from server.voice.tts_session import TtsSession


def make_settings(**overrides) -> Settings:
    return dataclasses.replace(Settings(), **overrides)


def make_stores(data_dir: str):
    settings = make_settings(data_dir=data_dir, history_db_path="")
    index = IndexStore(settings)
    index.open()
    media = MediaStore(settings, index)
    media.open()
    recorder = HistoryRecorder(settings, index)
    recorder.open()
    return settings, index, media, recorder


async def make_engines() -> EngineSet:
    tts = TtsSession(FakeTtsEngine(), "test", lambda payload: None)
    return EngineSet(vlm=FakeVlmSession(), asr=FakeAsrAdapter(), tts=tts)


def jpeg_with_exif() -> bytes:
    img = Image.new("RGB", (320, 200), (200, 30, 60))
    exif = Image.Exif()
    exif[271] = "SecretCameraMake"  # Make
    exif[272] = "SecretModel"       # Model
    buf = BytesIO()
    img.save(buf, format="JPEG", exif=exif.tobytes())
    return buf.getvalue()


# ------------------------------------------------------------------ realtime recording


async def test_realtime_recording(data_dir: str) -> str:
    settings, index, _media, recorder = make_stores(data_dir)
    manager = SessionManager(settings, history=recorder)
    state = await manager.create(SessionConfig(), make_engines)
    cid = state.session_id

    # emit the semantic stream the orchestrator would produce
    state.emit("input.transcription.done", text="画面里是什么东西", item_id="item_1",
               auto=False, asr_ms=123.4)
    state.emit("input.text.done", text="please type an answer", item_id="item_2")
    state.emit("response.created", response_id="resp_1")
    state.emit("response.text.delta", response_id="resp_1", delta="这是")  # NOT journaled
    state.emit("response.text.done", response_id="resp_1", text="这是一个流式测试回答")
    state.emit("response.done", response_id="resp_1", stop_reason="end_turn", ttft_ms=456.7)
    state.emit("status", transient=True)  # transient → never recorded

    await manager.close(cid, reason="client")
    recorder.close()  # flush the writer queue

    # journal: exists, header first, delta excluded
    files = glob.glob(os.path.join(data_dir, "journal", "*", "*", f"{cid}.jsonl"))
    assert len(files) == 1, files
    lines = [json.loads(x) for x in open(files[0], encoding="utf-8")]
    assert lines[0]["type"] == "history.open" and lines[0]["kind"] == "realtime"
    types = [x["type"] for x in lines]
    assert "response.text.delta" not in types, "deltas must not be journaled"
    assert types[-1] == "history.finalize"

    # index: conversation + 3 turns, ASR title, metrics preserved
    conv = index.get_conversation(cid)
    assert conv is not None and conv["kind"] == "realtime"
    assert conv["title"] == "画面里是什么东西", conv["title"]
    assert conv["ended_at"] is not None and conv["end_reason"] == "client"
    turns = index.get_transcript(cid)
    assert [t["role"] for t in turns] == ["user", "user", "assistant"], turns
    assert turns[0]["source"] == "asr" and json.loads(turns[0]["metrics_json"])["asr_ms"] == 123.4
    assert turns[1]["source"] == "typed"
    assert turns[2]["text"] == "这是一个流式测试回答"
    assert json.loads(turns[2]["metrics_json"])["ttft_ms"] == 456.7
    assert turns[2]["stop_reason"] == "end_turn"

    # search: FTS (≥3 chars), LIKE fallback (<3), and a miss
    assert index.list_conversations("流式测试")[0]["conversation_id"] == cid
    assert index.list_conversations("画面")[0]["conversation_id"] == cid  # 2 chars → LIKE
    assert index.list_conversations("不存在的词组") == []
    index.close()
    print("realtime recording: OK")
    return cid


# ------------------------------------------------------------------ media CAS + chat turns


def test_media_and_chat(data_dir: str) -> str:
    _settings, index, media, recorder = make_stores(data_dir)

    # ingest: EXIF stripped, deduped
    raw = jpeg_with_exif()
    desc = media.put_bytes(raw, "photo.jpg")
    h = desc["hash"]
    blob = media.blob_path(h)
    assert os.path.exists(blob) and desc["mime"] == "image/jpeg"
    assert desc["width"] == 320 and desc["height"] == 200
    stored_exif = Image.open(blob).getexif()
    assert len(stored_exif) == 0, f"EXIF must be stripped, got {dict(stored_exif)}"
    assert desc["thumb_path"] and os.path.exists(os.path.join(data_dir, desc["thumb_path"]))

    desc2 = media.put_bytes(raw, "photo-copy.jpg")  # same content → same identity
    assert desc2["hash"] == h, "dedup: identical bytes must hash identically"
    blobs = glob.glob(os.path.join(data_dir, "media", "blobs", "**", "*"), recursive=True)
    assert sum(1 for b in blobs if os.path.isfile(b)) == 1, "dedup: one stored copy"

    # rejects: garbage magic bytes; disallowed type (gif sniffs but isn't allowed)
    for payload, want in ((b"this is not an image " * 10, 415),
                          (b"GIF89a" + b"\x00" * 64, 415)):
        try:
            media.put_bytes(payload)
        except MediaError as exc:
            assert exc.status == want, (payload[:12], exc.status)
        else:
            raise AssertionError("bad payload must be rejected")

    # video plumbing: valid ftyp header → stored as video/mp4 (probe degrades)
    fake_mp4 = b"\x00\x00\x00\x20ftypisom" + b"\x00" * 256
    vdesc = media.put_bytes(fake_mp4, "clip.mp4")
    assert vdesc["kind"] == "video" and vdesc["mime"] == "video/mp4"
    assert os.path.exists(media.blob_path(vdesc["hash"]))

    # handle normalization
    assert normalize_hash(f"sha256:{h}") == h and normalize_hash(h) == h
    assert normalize_hash("data:image/jpeg;base64,xxx") is None

    # chat thread with the image attached
    cid = "chat_thread_0001"
    recorder.open_conversation(cid, "chat")
    recorder.record_turn(cid, role="user", text="describe the attached photo",
                         media_hashes=[h, "0" * 64])  # unknown hash is skipped
    recorder.record_turn(cid, role="assistant", text="a red rectangle", source="vlm",
                         metrics={"latency_ms": 88.0})
    recorder.close()

    conv = index.get_conversation(cid)
    assert conv and conv["kind"] == "chat" and conv["title"] == "describe the attached photo"
    turns = index.get_transcript(cid)
    assert len(turns) == 2 and len(turns[0]["media"]) == 1
    assert turns[0]["media"][0]["hash"] == h
    assert index.get_media(h)["ref_count"] == 1
    index.close()
    print("media + chat recording: OK")
    return cid


# ------------------------------------------------------------------ file-streaming session


async def test_video_session(data_dir: str) -> str:
    """video.attach turn + per-turn media_ts anchors + the kind facet."""
    settings, index, media, recorder = make_stores(data_dir)
    # same bytes as test_media_and_chat's clip → CAS dedup keeps one blob
    fake_mp4 = b"\x00\x00\x00\x20ftypisom" + b"\x00" * 256
    vhash = media.put_bytes(fake_mp4, "clip.mp4")["hash"]

    manager = SessionManager(settings, history=recorder)
    state = await manager.create(SessionConfig(video_source="file"), make_engines)
    cid = state.session_id

    # the stream the orchestrator produces for a file session: attach first,
    # then turns stamped with the video position they happened at
    state.emit("input.video.attached", media=f"sha256:{vhash}", name="clip.mp4",
               duration_s=12.5, item_id="item_1")
    state.emit("input.text.done", text="what happens here", item_id="item_2", media_ts=3.2)
    state.emit("response.created", response_id="resp_1")
    state.emit("response.text.done", response_id="resp_1", text="a title card appears")
    state.emit("response.done", response_id="resp_1", stop_reason="end_turn", media_ts=3.2)

    await manager.close(cid, reason="client")
    recorder.close()

    conv = index.get_conversation(cid)
    assert conv is not None and conv["kind"] == "realtime"
    assert json.loads(conv["config_json"])["video_source"] == "file"
    assert conv["title"] == "clip.mp4"  # the attach names the session (no earlier user turn)

    turns = index.get_transcript(cid)
    assert [t["source"] for t in turns] == ["video", "typed", "vlm"], turns
    assert turns[0]["media"][0]["hash"] == vhash and turns[0]["media"][0]["kind"] == "video"
    assert json.loads(turns[0]["metrics_json"])["duration_s"] == 12.5
    assert json.loads(turns[1]["metrics_json"])["media_ts"] == 3.2
    assert json.loads(turns[2]["metrics_json"])["media_ts"] == 3.2
    assert index.get_media(vhash)["ref_count"] >= 1, "attach must ref-count the blob"

    # kind facet: the live sidebar lists realtime only, the chat sidebar chat only
    assert {c["kind"] for c in index.list_conversations(kind="realtime")} == {"realtime"}
    assert all(c["kind"] == "chat" for c in index.list_conversations(kind="chat"))
    assert index.list_conversations("title card", kind="chat") == []
    assert index.list_conversations("title card", kind="realtime")[0]["conversation_id"] == cid
    index.close()
    print("video session (attach + media_ts + kind facet): OK")
    return cid


# ------------------------------------------------------------------ delete + prune


def test_delete(data_dir: str, chat_cid: str) -> None:
    settings = make_settings(data_dir=data_dir, history_db_path="")
    index = IndexStore(settings)
    index.open()
    media = MediaStore(settings, index)
    recorder = HistoryRecorder(settings, index)  # delete path needs no writer thread

    h = index.get_transcript(chat_cid)[0]["media"][0]["hash"]
    assert recorder.delete_conversation(chat_cid) is True
    assert index.get_conversation(chat_cid) is None
    assert glob.glob(os.path.join(data_dir, "journal", "*", "*", f"{chat_cid}.jsonl")) == []
    assert index.get_media(h)["ref_count"] == 0, "delete must release refs"
    assert any(m["hash"] == h for m in index.unreferenced_media())

    media.delete_blob(h)  # what the prune tool does
    index.delete_media_row(h)
    assert not os.path.exists(media.blob_path(h)) and index.get_media(h) is None
    index.close()
    print("delete + release refs: OK")


# ------------------------------------------------------------------ rebuild from journals


def test_rebuild(data_dir: str, realtime_cid: str, chat_cid: str, video_cid: str) -> None:
    settings = make_settings(data_dir=data_dir, history_db_path="")
    db = os.path.join(data_dir, "index.db")

    # sabotage: corrupt tail on one journal (crash mid-append), then drop the DB
    journal = glob.glob(os.path.join(data_dir, "journal", "*", "*", f"{realtime_cid}.jsonl"))[0]
    with open(journal, "a", encoding="utf-8") as f:
        f.write('{"type":"response.done","resp')  # truncated line
    for suffix in ("", "-wal", "-shm"):
        if os.path.exists(db + suffix):
            os.unlink(db + suffix)

    index = IndexStore(settings)
    index.open()
    media = MediaStore(settings, index)
    media.open()
    assert media.rescan_blobs() == 2  # image + fake video
    for path in iter_journal_files(os.path.join(data_dir, "journal")):
        replay_file(index, path)

    conv = index.get_conversation(realtime_cid)
    assert conv and conv["kind"] == "realtime" and conv["title"] == "画面里是什么东西"
    assert [t["role"] for t in index.get_transcript(realtime_cid)] == ["user", "user", "assistant"]
    chat = index.get_conversation(chat_cid)
    turns = index.get_transcript(chat_cid)
    assert chat and len(turns) == 2 and len(turns[0]["media"]) == 1
    assert index.get_media(turns[0]["media"][0]["hash"])["ref_count"] == 1
    assert index.list_conversations("流式测试")[0]["conversation_id"] == realtime_cid
    # the file-streaming extras survive a rebuild too
    vturns = index.get_transcript(video_cid)
    assert vturns[0]["source"] == "video" and len(vturns[0]["media"]) == 1
    assert json.loads(vturns[1]["metrics_json"])["media_ts"] == 3.2
    index.close()
    print("rebuild from journals + blob rescan: OK")


# ------------------------------------------------------------------ multi-turn chat prep


def test_prepare_chat_messages(data_dir: str) -> None:
    """vlm_hf._prepare_chat_messages: multi-turn parts, CAS handles, order, legacy media."""
    import base64

    from server.adapters.vlm.moss_vl_hf.adapter import HfMossVlAdapter
    from server.schemas import ChatRequest

    _settings, index, media, _recorder = make_stores(data_dir)
    set_media_store(media)
    try:
        # CAS image: 320x200 (from jpeg_with_exif); inline image: 64x48
        h = media.put_bytes(jpeg_with_exif(), "ctx.jpg")["hash"]
        small = BytesIO()
        Image.new("RGB", (64, 48), (10, 200, 10)).save(small, format="JPEG")
        data_url = "data:image/jpeg;base64," + base64.b64encode(small.getvalue()).decode()
        fake_mp4 = b"\x00\x00\x00\x20ftypisom" + b"\x00" * 256
        vh = media.put_bytes(fake_mp4, "clip.mp4")["hash"]
        vblob = media.blob_path(vh)

        req = ChatRequest(**{
            "messages": [
                {"role": "user", "content": [
                    {"type": "image", "media": f"sha256:{h}"},
                    {"type": "text", "text": "turn one"}]},
                {"role": "assistant", "content": "reply one"},
                {"role": "user", "content": [
                    {"type": "video", "media": f"sha256:{vh}"},
                    {"type": "text", "text": "turn two"}]},
            ],
            "images": [data_url],  # legacy top-level → attaches to LAST user msg
        })
        messages, images, videos = HfMossVlAdapter._prepare_chat_messages(req)

        # document order: msg1's CAS image (320x200) first, then msg3's inline (64x48)
        assert [im.size for im in images] == [(320, 200), (64, 48)], [im.size for im in images]
        # video handles resolve to blob PATHS (torchcodec decodes the file itself)
        assert videos == [{"video_path": vblob}], videos
        assert messages[0]["content"] == [{"type": "image"}, {"type": "text", "text": "turn one"}]
        assert messages[1]["content"] == "reply one"  # untouched string
        assert messages[2]["content"] == [
            {"type": "image"}, {"type": "video"}, {"type": "text", "text": "turn two"}]

        # legacy top-level videos attach to the LAST user message too
        req_v = ChatRequest(**{
            "messages": [{"role": "user", "content": "describe"}],
            "videos": [f"sha256:{vh}"]})
        messages_v, _, videos_v = HfMossVlAdapter._prepare_chat_messages(req_v)
        assert videos_v == [{"video_path": vblob}]
        assert messages_v[0]["content"] == [
            {"type": "video"}, {"type": "text", "text": "describe"}]

        # videos accept CAS handles ONLY — raw paths/base64 must be rejected
        for bad in ("/etc/passwd", "data:video/mp4;base64,AAAA", "nonsense"):
            try:
                HfMossVlAdapter._prepare_chat_messages(ChatRequest(**{"messages": [
                    {"role": "user", "content": [{"type": "video", "media": bad}]}]}))
                raise AssertionError(f"accepted non-handle video payload: {bad!r}")
            except ValueError:
                pass

        # text-only multi-turn: no media, strings pass through
        req2 = ChatRequest(**{"messages": [
            {"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "again"}]})
        messages2, images2, videos2 = HfMossVlAdapter._prepare_chat_messages(req2)
        assert images2 == [] and videos2 == []
        assert [m["content"] for m in messages2] == ["hi", "hello", "again"]
    finally:
        set_media_store(None)
        index.close()
    print("prepare chat messages (multi-turn): OK")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="hist_rt_") as d1:
        asyncio.run(test_realtime_recording(d1))

    with tempfile.TemporaryDirectory(prefix="hist_prep_") as d3:
        test_prepare_chat_messages(d3)

    # media/chat, then rebuild (same dir), then delete (same dir)
    with tempfile.TemporaryDirectory(prefix="hist_media_") as d2:
        chat_cid = test_media_and_chat(d2)

        # need realtime journals alongside for the rebuild scenario
        realtime_cid = asyncio.run(test_realtime_recording(d2))
        video_cid = asyncio.run(test_video_session(d2))
        test_rebuild(d2, realtime_cid, chat_cid, video_cid)
        test_delete(d2, chat_cid)

    print("\nPERSISTENCE TEST OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
