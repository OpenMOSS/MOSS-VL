"""L2 memory: language policy, store/vectors, retrieval gate, injection format,
re-injection budgets, z-norm, late-interaction blobs, fact re-keying, rewriting.

    .venv/bin/python -m server.tests.test_memory

Runs with no GPU, no model weights and no network — the embedders fall back to
the deterministic hashing/descriptor pair, which is exactly the configuration a
fresh box boots in.
"""
from __future__ import annotations

import asyncio
import io
import os
import tempfile
import time

import numpy as np

from .. import config as config_mod
from ..config import Settings
from ..memory import inject as inj
from ..memory import lang
from ..memory import rewrite
from ..memory.embed import HashingTextEmbedder, TinyImageEmbedder, dhash, hamming
from ..memory.facts import FactExtractor, parse_facts
from ..memory.retrieval import Retriever, _minmax
from ..memory.session import MemorySession
from ..memory.store import KIND_UTTERANCE, SPACE_TEXT, SPACE_TEXT_LI, MemoryStore
from ..memory.writer import MemoryWriter
from .fakes import FakeOfflineVlm


def _settings(tmp: str, **over):
    os.environ["DATA_DIR"] = tmp
    os.environ["MEMORY_ENABLED"] = "1"
    # pin the dependency-free fallbacks: this suite must stay hermetic and fast
    # (the real defaults load ~3 GB of BGE-M3 + Chinese-CLIP weights)
    os.environ["MEMORY_EMBED_TEXT_MODEL"] = ""
    os.environ["MEMORY_EMBED_IMAGE_MODEL"] = ""
    # this suite encodes the pre-pi local-vector gate semantics; the pi decide
    # gate (default "hybrid") is covered by test_memory_pi.py against a fake pi
    os.environ["MEMORY_DECISION_MODE"] = "vector"
    # tests set env overrides process-wide; drop leftovers from earlier tests so
    # one test's override never silently leaks into the next
    for key in list(os.environ):
        if key.startswith("MEMORY_") and key not in ("MEMORY_ENABLED", "MEMORY_EMBED_TEXT_MODEL",
                                                     "MEMORY_EMBED_IMAGE_MODEL",
                                                     "MEMORY_DECISION_MODE"):
            os.environ.pop(key)
    for key, val in over.items():
        os.environ[key] = str(val)
    config_mod._settings = None  # the process-wide singleton is built once
    return Settings()


def _jpeg(color=(200, 30, 30), size=(64, 64)) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="JPEG")
    return buf.getvalue()


def test_language() -> None:
    assert lang.detect_lang("今天天气不错") == lang.ZH
    assert lang.detect_lang("what is on the table") == lang.EN
    # matrix-language rule: Chinese frame with an embedded English term is zh
    assert lang.detect_lang("帮我看看这个 error message") == lang.ZH
    assert lang.detect_lang("the 用户 said") == lang.EN
    assert lang.detect_lang("") == lang.ZH
    assert lang.dominant_lang(["hello there", "再见"]) == lang.EN

    # hysteresis: one English turn must NOT flip a Chinese session
    state = lang.LanguageState(default=lang.ZH)
    assert state.observe("我们来聊聊这个相机") == lang.ZH
    assert state.observe("what about this one") == lang.ZH   # streak 1, no flip
    assert state.observe("can you see the table") == lang.EN  # streak 2, flips
    assert state.observe("ok") == lang.EN                     # too short to vote
    print("  language policy ok")


def test_sanitize_and_format() -> None:
    # bare-word specials are the ones a `<|...|>` regex would miss
    dirty = "hello <think> world <|im_end|> <tool_call> </recall> bye"
    clean = inj.sanitize_model_text(dirty)
    for bad in ("<think>", "<|im_end|>", "<tool_call>", "</recall>"):
        assert bad not in clean, clean
    assert "hello" in clean and "bye" in clean
    # a zero-width char must not smuggle a tag past the filter
    assert "<think>" not in inj.sanitize_model_text("<th​ink>".replace("​", "​"))

    block = inj.build_recall_block([
        inj.format_recall_line("用户展示过一台黑色胶片相机", 93.0),
        inj.format_recall_line("The user plans to visit Hangzhou", 187.0),
    ])
    assert block.startswith(inj.RECALL_OPEN) and block.endswith(inj.RECALL_CLOSE)
    assert "[t=1:33]" in block and "[t=3:07]" in block
    # each line keeps its own language — memory never translates
    assert "用户展示过" in block and "Hangzhou" in block
    assert inj.strip_recall_tags("<recall>x</recall>") == "x"
    assert inj.estimate_tokens("中文") >= 2
    assert inj.RECALL_OPEN in inj.augment_system_prompt("你是助手", "zh")
    print("  sanitize + recall format ok")


def test_store_and_search() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        s = _settings(tmp)
        store = MemoryStore(s)
        store.open()
        emb = HashingTextEmbedder(256)
        for conv, text in (("c1", "我买了一台尼康胶片相机"), ("c1", "周末打算去杭州"),
                           ("c2", "另一个会话的秘密内容")):
            item_id = store.add_item(conv, KIND_UTTERANCE, text=text, role="user", session_ts=1.0)
            store.add_vector(conv, item_id, SPACE_TEXT, emb.encode([text])[0])

        hits = store.search("c1", SPACE_TEXT, emb.encode(["胶片相机"])[0], limit=3)
        assert hits, "expected a lexical hit"
        top = store.get_items([hits[0][0]])[hits[0][0]]
        assert "相机" in top.text

        # isolation: c1's query must never reach c2's rows
        other = store.search("c1", SPACE_TEXT, emb.encode(["另一个会话的秘密内容"])[0], limit=5)
        assert all(store.get_items([i])[i].conversation_id == "c1" for i, _ in other)
        assert store.count("c2") == 1
        store.close()
    print("  store + per-session isolation ok")


def test_frame_dedup() -> None:
    a, b = _jpeg((200, 30, 30)), _jpeg((201, 31, 31))
    c = _jpeg((10, 200, 40))
    assert hamming(dhash(a), dhash(b)) <= 6, "near-identical frames must collapse"
    img = TinyImageEmbedder()
    vecs = img.encode_images([a, b, c])
    assert float(vecs[0] @ vecs[1]) > float(vecs[0] @ vecs[2])
    print("  frame dedup ok")


def test_recall_gate_and_budget() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        s = _settings(tmp, MEMORY_INJECT_MAX_ITEMS=2, MEMORY_INJECT_MAX_TOKENS=120)
        store = MemoryStore(s)
        writer = MemoryWriter(s, store)
        writer.start()
        sess = MemorySession("sess-1", s, store, writer)

        sess.note_user_turn("我刚买了一台尼康 FM2 胶片相机，很喜欢")
        sess.note_user_turn("周末打算去杭州出差")
        sess.note_frame(_jpeg())
        writer.drain()

        # 1. a turn with no past-reference and no lexical overlap → no injection
        assert not sess.recall_for_turn("现在几点了"), "gate should skip an unrelated turn"

        # 2. an explicit past-reference with real overlap → injected, formatted
        got = sess.recall_for_turn("我刚才说的那台胶片相机是什么型号")
        assert got, "explicit past reference with overlap should recall"
        assert inj.RECALL_OPEN in got.block
        assert inj.estimate_tokens(got.block) <= s.memory_inject_max_tokens + 8
        assert len(got.ids) <= s.memory_inject_max_items

        # 3. an immediate re-ask is dropped by the distance gate (design §5),
        #    not by up-front exclusion — see test_reinjection for the full cycle
        sess.mark_injected(got.ids)
        again = sess.recall_for_turn("我刚才说的那台胶片相机是什么型号")
        assert all(i not in got.ids for i in again.ids), "re-injected inside the distance gate"

        assert sess.status()["items"] >= 3
        writer.stop()
        store.close()
    print("  recall gate + budget + distance gate ok")


def _make_session(tmp: str, conv: str = "sess-1", **over):
    s = _settings(tmp, **over)
    store = MemoryStore(s)
    writer = MemoryWriter(s, store)
    writer.start()
    return s, store, writer, MemorySession(conv, s, store, writer)


def test_reinjection() -> None:
    """Distance gate, copy cap, short form — design §5 re-injection."""
    with tempfile.TemporaryDirectory() as tmp:
        s, store, writer, sess = _make_session(tmp, MEMORY_REINJECT_DISTANCE=40)
        sess.note_user_turn("我刚买了一台尼康 FM2 胶片相机。它几乎全新，花了两千块。")
        writer.drain()
        query = "我刚才说的那台胶片相机是什么型号"

        first = sess.recall_for_turn(query)
        assert first, "first ask should recall"
        sess.mark_injected(first.ids)

        # still inside the effective window: distance < 40 → dropped
        assert not sess.recall_for_turn(query), "re-injection inside the distance gate"

        # push the internal token estimate past the distance with an unrelated turn
        sess.note_user_turn("今天天气怎么样我们聊点完全不相干的话题吧说说别的事情")
        writer.drain()
        second = sess.recall_for_turn(query)
        assert second and set(second.ids) & set(first.ids), "past the distance it must re-inject"
        # short form: first clause only, not a byte-repeat of the whole turn
        assert "尼康" in second.block
        assert "它几乎全新" not in second.block
        sess.mark_injected(second.ids)
        assert sess._injected[first.ids[0]].copies == 2
        writer.stop()
        store.close()

    with tempfile.TemporaryDirectory() as tmp:
        # copy cap: distance 0 disables the time half of the gate
        s, store, writer, sess = _make_session(
            tmp, MEMORY_REINJECT_DISTANCE=0, MEMORY_REINJECT_MAX_COPIES=3)
        sess.note_user_turn("我刚买了一台尼康 FM2 胶片相机")
        writer.drain()
        query = "我刚才说的那台胶片相机是什么型号"
        for _ in range(3):
            got = sess.recall_for_turn(query)
            assert got
            sess.mark_injected(got.ids)
        assert not sess.recall_for_turn(query), "a 4th copy must be dropped by the cap"
        writer.stop()
        store.close()
    print("  re-injection distance/copies/short-form ok")


def test_lifetime_budget() -> None:
    """Lifetime injected-token exhaustion stops recall SILENTLY (rollover signal)."""
    with tempfile.TemporaryDirectory() as tmp:
        s, store, writer, sess = _make_session(
            tmp, MEMORY_REINJECT_DISTANCE=0, MEMORY_INJECT_SESSION_MAX_TOKENS=30)
        sess.note_user_turn("我刚买了一台尼康 FM2 胶片相机，很喜欢")
        writer.drain()
        query = "我刚才说的那台胶片相机是什么型号"
        first = sess.recall_for_turn(query)
        assert first
        sess.mark_injected(first.ids)
        assert 0 < sess._lifetime_tokens <= 30
        assert not sess.recall_for_turn(query), "lifetime budget exhausted but recall continued"
        writer.stop()
        store.close()
    print("  lifetime budget ok")


def test_note_rollover() -> None:
    """Injected table recomputed from the new prefix; suppression and the
    lifetime budget persist; distances reset."""
    with tempfile.TemporaryDirectory() as tmp:
        conv = "sess-roll"
        s, store, writer, sess = _make_session(tmp, conv=conv, MEMORY_REINJECT_DISTANCE=50)
        sess.note_user_turn("我刚买了一台尼康 FM2 胶片相机")
        sess.note_user_turn("周末打算去杭州出差")
        writer.drain()
        query = "我刚才说的那台胶片相机是什么型号"
        first = sess.recall_for_turn(query)
        assert first
        sess.mark_injected(first.ids)
        other = [i.id for i in store.recent(conv, [KIND_UTTERANCE]) if i.id not in first.ids]
        sess.suppress(other)
        lifetime_before = sess._lifetime_tokens

        sess.note_rollover(list(first.ids), text_tokens=100.0)
        assert set(sess._injected) == set(first.ids), "table must be EXACTLY the kept ids"
        entry = sess._injected[first.ids[0]]
        assert entry.copies == 1 and entry.last_tokens == 100.0
        assert all(i in sess._suppressed for i in other), "suppression must survive rollover"
        assert sess._lifetime_tokens == lifetime_before, "lifetime budget must NOT reset"

        # distances reset: same position → still gated; +100 tokens → re-injectable
        assert not sess.recall_for_turn(query, now_tokens=100.0)
        again = sess.recall_for_turn(query, now_tokens=200.0)
        assert again and set(again.ids) & set(first.ids)
        writer.stop()
        store.close()
    print("  note_rollover ok")


def test_znorm_fallback() -> None:
    """Below 30 background samples a space keeps the min-max behaviour."""
    with tempfile.TemporaryDirectory() as tmp:
        s = _settings(tmp)
        store = MemoryStore(s)
        store.open()
        emb = HashingTextEmbedder(256)
        retr = Retriever(store, emb, TinyImageEmbedder(), s)
        vals = [0.5, 0.7, 0.9]
        retr._observe_scores(SPACE_TEXT, [0.42] * 10)
        assert retr._normalize(SPACE_TEXT, vals) == _minmax(vals), "<30 samples must min-max"
        bg = [0.3 + 0.01 * i for i in range(40)]
        retr._observe_scores(SPACE_TEXT, bg)
        arr = np.asarray([0.42] * 10 + bg)
        expected = [(v - arr.mean()) / arr.std() for v in vals]
        got = retr._normalize(SPACE_TEXT, vals)
        assert all(abs(a - b) < 1e-6 for a, b in zip(got, expected)), "z-norm mismatch"
        store.close()
    print("  z-norm fallback ok")


def test_update_vector_no_duplicate() -> None:
    """Fact re-keying replaces the in-RAM row in place (add_vector would dup)."""
    with tempfile.TemporaryDirectory() as tmp:
        s = _settings(tmp)
        store = MemoryStore(s)
        store.open()
        emb = HashingTextEmbedder(256)
        item = store.add_item("c1", KIND_UTTERANCE, text="我买了一台尼康相机",
                              role="user", session_ts=1.0)
        store.add_vector("c1", item, SPACE_TEXT, emb.encode(["完全无关的旧向量内容"])[0])
        store.search("c1", SPACE_TEXT, emb.encode(["相机"])[0], limit=3)  # load the RAM index
        store.update_vector("c1", item, SPACE_TEXT, emb.encode(["我买了一台尼康相机"])[0])
        idx = store._idx[("c1", SPACE_TEXT)]
        assert idx.ids.count(item) == 1, "update_vector must replace, not duplicate"
        hits = store.search("c1", SPACE_TEXT, emb.encode(["我买了一台尼康相机"])[0], limit=3)
        assert hits and hits[0][0] == item and hits[0][1] > 0.99, "stale vector survived rekey"
        store.put_key(item, "我买了一台尼康相机 尼康 相机")
        assert store.get_key(item) == "我买了一台尼康相机 尼康 相机"
        store.close()
    print("  update_vector no-duplicate ok")


def test_late_store() -> None:
    """Variable-length text_li blobs: pack/load round-trip, maxsim, in-place replace."""
    with tempfile.TemporaryDirectory() as tmp:
        s = _settings(tmp)
        store = MemoryStore(s)
        store.open()
        item = store.add_item("c1", KIND_UTTERANCE, text="x", role="user")
        rng = np.random.default_rng(0)
        mat = rng.normal(size=(5, 8)).astype(np.float32)
        mat /= np.linalg.norm(mat, axis=1, keepdims=True)
        store.add_vector_late("c1", item, mat)
        q = rng.normal(size=(3, 8)).astype(np.float32)
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        hits = store.search_late("c1", q, limit=4)
        assert hits and hits[0][0] == item
        expected = float((mat @ q.T).max(axis=0).mean())
        assert abs(hits[0][1] - expected) < 1e-5, "maxsim mean mismatch"
        mat2 = rng.normal(size=(7, 8)).astype(np.float32)
        store.update_vector("c1", item, SPACE_TEXT_LI, mat2)
        assert len(store._li["c1"]) == 1 and store._li["c1"][item].shape == (7, 8)
        store.close()
    print("  late-interaction store ok")


def test_facts_rekey() -> None:
    """Facts enrich the index key only; the stored turn stays raw and verbatim."""
    with tempfile.TemporaryDirectory() as tmp:
        conv = "sess-facts"
        s, store, writer, sess = _make_session(tmp, conv=conv)
        fake = FakeOfflineVlm(reply=("- 用户有一台尼康 FM2\n", "- 相机是胶片机"))
        extractor = FactExtractor(s, store, writer, plane=fake)
        sess.note_user_turn("我刚买的尼康 FM2 真好用")
        writer.drain()
        item = store.recent(conv, [KIND_UTTERANCE])[0]

        assert parse_facts("- 事实一\n1. 事实二\n\n") == ["事实一", "事实二"]
        key = asyncio.run(extractor.extract_and_rekey(
            conv, item.id, "我刚买的尼康 FM2 真好用", ["助手: 买了什么？"]))
        assert key and "胶片机" in key and key.startswith("我刚买的尼康 FM2 真好用")
        assert store.get_key(item.id) == key
        # the model never sees facts: the stored row is still the raw turn
        assert store.get_items([item.id])[item.id].text == "我刚买的尼康 FM2 真好用"
        # re-embedded on the key: a fact-only term now retrieves the turn
        hits = store.search(conv, SPACE_TEXT, writer.text.encode(["胶片机"])[0], limit=3)
        assert hits and hits[0][0] == item.id

        # silent no-ops: provider off, plane unloaded
        s_none = _settings(tmp, MEMORY_SUMMARY_PROVIDER="none")
        before = fake.calls
        assert asyncio.run(FactExtractor(s_none, store, writer, plane=fake)
                           .extract_and_rekey(conv, item.id, "我刚买的尼康 FM2 真好用")) is None
        assert asyncio.run(FactExtractor(s, store, writer, plane=FakeOfflineVlm(loaded=False))
                           .extract_and_rekey(conv, item.id, "我刚买的尼康 FM2 真好用")) is None
        assert fake.calls == before, "no-op paths must never touch the plane"

        # e2e through the session facade: item id resolved by polling the store
        sess._facts = extractor
        asyncio.run(sess.maybe_extract_facts(None, "我刚买的尼康 FM2 真好用", None))
        assert fake.calls > before
        writer.stop()
        store.close()
    print("  facts rekey ok")


def test_rewrite_rules() -> None:
    now = 600.0
    w = rewrite.parse_time_window("你3分钟前看到的那个是什么", now)
    assert w and w[0] <= 420.0 <= w[1]
    assert rewrite.parse_time_window("刚才那个东西是什么", now) == (300.0, 600.0)
    assert rewrite.parse_time_window("看 t=1:30 那里", now) == (60.0, 120.0)
    w = rewrite.parse_time_window("what did I show you 5 minutes ago", now)
    assert w and w[0] <= 300.0 <= w[1]
    w = rewrite.parse_time_window("what happened 30 seconds ago", now)
    assert w and w[0] <= 570.0 <= w[1]
    assert rewrite.parse_time_window("今天天气怎么样", now) is None
    assert rewrite.parse_time_window("3分钟前说过", 100.0) is None  # before session start
    # session-start references restrict to the first minute of the session
    assert rewrite.parse_time_window("我最开始说的一句话是什么", now) == (0.0, 60.0)
    assert rewrite.parse_time_window("一开始那个人戴了什么", now) == (0.0, 60.0)
    assert rewrite.parse_time_window("what was the first thing I said", now) == (0.0, 60.0)
    assert rewrite.parse_time_window("我最开始说了什么", 30.0) == (0.0, 30.0)  # young session

    assert rewrite.is_deictic_query("刚才那个是什么")
    assert rewrite.is_deictic_query("what was that one earlier")
    assert not rewrite.is_deictic_query("现在几点了")
    long_q = "请详细介绍一下尼康 FM2 胶片相机的历史背景以及它和 FM3a 的全部区别那个"
    assert not rewrite.is_deictic_query(long_q), "long queries carry their own content"
    aug = rewrite.augment_query("刚才那个是什么", ["我买了一台相机"])
    assert aug.startswith("刚才那个是什么") and "相机" in aug
    assert rewrite.augment_query("刚才那个是什么", []) == "刚才那个是什么"
    print("  rewrite rules ok")


def test_time_window_filter() -> None:
    """A confident relative-time parse restricts candidates by session_ts."""
    with tempfile.TemporaryDirectory() as tmp:
        s, store, writer, sess = _make_session(tmp)
        sess.note_user_turn("我想买一台佳能相机", session_ts=30.0)
        sess.note_user_turn("我想买一台尼康相机", session_ts=420.0)
        writer.drain()
        sess._t0 = time.monotonic() - 600.0  # pretend the session is 10 min old
        got = sess.recall_for_turn("你记不记得我3分钟前说的那台相机")
        assert got and "尼康" in got.block and "佳能" not in got.block
        writer.stop()
        store.close()
    print("  time-window filter ok")


def main() -> None:
    test_language()
    test_sanitize_and_format()
    test_store_and_search()
    test_frame_dedup()
    test_recall_gate_and_budget()
    test_reinjection()
    test_lifetime_budget()
    test_note_rollover()
    test_znorm_fallback()
    test_update_vector_no_duplicate()
    test_late_store()
    test_facts_rekey()
    test_rewrite_rules()
    test_time_window_filter()
    print("memory: all checks passed")


if __name__ == "__main__":
    main()
