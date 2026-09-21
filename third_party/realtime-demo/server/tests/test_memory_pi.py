"""pi_agent memory integration (MIGRATION_PLAN §2): the decide gate (vector /
llm / hybrid + retro loosening + unreachable degradation), the "pi" compact
provider (summary + pins into the prefix), rollover compact prefetch (hit /
in-flight join / failure degrade), writer utterance dedup, and the
interrupted-turn commit=False rule.

    .venv/bin/python -m server.tests.test_memory_pi

Hermetic like test_memory.py: tmp DATA_DIR, fallback embedders via empty model
paths, and a FAKE pi_agent (stdlib http.server on a thread, loopback only).
No GPU, no external network.
"""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .. import config as config_mod
from ..config import Settings
from ..memory.rollover import RolloverManager
from ..memory.session import MemorySession
from ..memory.store import KIND_PINNED, KIND_UTTERANCE, MemoryStore
from ..memory.writer import MemoryWriter


def _settings(tmp: str, **over) -> Settings:
    os.environ["DATA_DIR"] = tmp
    os.environ["MEMORY_ENABLED"] = "1"
    # pin the dependency-free fallbacks: this suite must stay hermetic and fast
    os.environ["MEMORY_EMBED_TEXT_MODEL"] = ""
    os.environ["MEMORY_EMBED_IMAGE_MODEL"] = ""
    for key in list(os.environ):
        if key.startswith("MEMORY_") and key not in ("MEMORY_ENABLED", "MEMORY_EMBED_TEXT_MODEL",
                                                     "MEMORY_EMBED_IMAGE_MODEL"):
            os.environ.pop(key)
    for key, val in over.items():
        os.environ[key] = str(val)
    config_mod._settings = None  # the process-wide singleton is built once
    return Settings()


def _set(settings: Settings, name: str, value) -> None:
    """Settings is a frozen dataclass; tests re-pin a field post-construction."""
    object.__setattr__(settings, name, value)


class _FakePiHandler(BaseHTTPRequestHandler):
    def log_message(self, *args) -> None:  # keep the test output clean
        pass

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        srv = self.server
        if self.path == "/decide":
            srv.decide_requests.append(body)
            resp = srv.decide_response
        elif self.path == "/compact":
            if srv.compact_delay:
                time.sleep(srv.compact_delay)
            srv.compact_requests.append(body)
            resp = srv.compact_response
        else:
            self.send_response(404)
            self.end_headers()
            return
        if resp is None:  # simulate a server-side failure (5xx)
            self.send_response(500)
            self.end_headers()
            return
        data = json.dumps(resp, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class FakePi:
    """Programmable fake pi_agent; records every request body."""

    def __init__(self, decide_response=None, compact_response=None, compact_delay: float = 0.0):
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _FakePiHandler)
        self.httpd.decide_requests = []
        self.httpd.compact_requests = []
        self.httpd.decide_response = (
            {"retrieve": True, "query": "", "reason": "fake"} if decide_response is None
            else decide_response)
        self.httpd.compact_response = (
            {"summary": "用户买了一台相机,预算两千元。", "pins": ["用户预算 2000 元"]}
            if compact_response is None else compact_response)
        self.httpd.compact_delay = compact_delay
        self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self._thread.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


def _make_session(tmp: str, conv: str = "c1", **over):
    s = _settings(tmp, **over)
    store = MemoryStore(s)
    writer = MemoryWriter(s, store)
    writer.start()
    return s, store, writer, MemorySession(conv, s, store, writer)


def test_decide_modes() -> None:
    pi = FakePi()
    try:
        # ---- vector: pi is never consulted, the local gate decides ----
        with tempfile.TemporaryDirectory() as tmp:
            s, store, writer, sess = _make_session(
                tmp, MEMORY_DECISION_MODE="vector", MEMORY_PI_URL=pi.url)
            sess.note_user_turn("我刚买了一台尼康 FM2 胶片相机，很喜欢")
            writer.drain()
            got = sess.recall_for_turn("我刚才说的那台胶片相机是什么型号")
            assert got, "vector mode must recall independently of pi"
            assert not pi.httpd.decide_requests, "vector mode must never call /decide"
            writer.stop()
            store.close()

        # ---- llm: every turn asks /decide; retrieve=false skips recall ----
        with tempfile.TemporaryDirectory() as tmp:
            s, store, writer, sess = _make_session(
                tmp, MEMORY_DECISION_MODE="llm", MEMORY_PI_URL=pi.url)
            sess.note_user_turn("我刚买了一台尼康 FM2 胶片相机，很喜欢")
            writer.drain()
            pi.httpd.decide_response = {"retrieve": False, "query": "", "reason": "no"}
            got = sess.recall_for_turn("我刚才说的那台胶片相机是什么型号")
            assert not got, "llm retrieve=false must skip recall"
            assert len(pi.httpd.decide_requests) == 1
            req = pi.httpd.decide_requests[-1]
            assert req["conversation_id"] == "c1"
            assert req["pending_user_text"] == "我刚才说的那台胶片相机是什么型号"
            assert "尼康" in req["recent_turns"], "recent_turns ride the /decide body"
            pi.httpd.decide_response = {"retrieve": True, "query": "", "reason": "yes"}
            got = sess.recall_for_turn("我刚才说的那台胶片相机是什么型号")
            assert got, "llm retrieve=true must recall"
            writer.stop()
            store.close()

        # ---- llm: the decide query rewrite overrides the search query ----
        with tempfile.TemporaryDirectory() as tmp:
            s, store, writer, sess = _make_session(
                tmp, MEMORY_DECISION_MODE="llm", MEMORY_PI_URL=pi.url)
            sess.note_user_turn("我上周买了一台尼康胶片相机")
            writer.drain()
            pi.httpd.decide_response = {"retrieve": True, "query": "尼康胶片相机", "reason": "rw"}
            got = sess.recall_for_turn("我之前说的那是什么东西")
            assert got and "尼康" in got.block, "decide query override must drive the search"
            writer.stop()
            store.close()

        # ---- hybrid: stage-1 prefilter first, /decide only confirms ----
        with tempfile.TemporaryDirectory() as tmp:
            s, store, writer, sess = _make_session(
                tmp, MEMORY_DECISION_MODE="hybrid", MEMORY_PI_URL=pi.url)
            sess.note_user_turn("我刚买了一台尼康 FM2 胶片相机，很喜欢")
            writer.drain()
            query = "尼康 FM2 胶片相机是什么型号"
            before = len(pi.httpd.decide_requests)
            _set(s, "memory_retrieval_prefilter_score", 2.0)  # above any cosine
            assert not sess.recall_for_turn(query), "below the prefilter: no recall"
            assert len(pi.httpd.decide_requests) == before, "below the prefilter: no /decide"
            _set(s, "memory_retrieval_prefilter_score", 0.01)
            pi.httpd.decide_response = {"retrieve": False, "query": "", "reason": "no"}
            assert not sess.recall_for_turn(query), "decide retrieve=false must veto"
            assert len(pi.httpd.decide_requests) == before + 1, "past the prefilter: /decide ran"
            pi.httpd.decide_response = {"retrieve": True, "query": "", "reason": "yes"}
            assert sess.recall_for_turn(query), "prefilter + decide pass must recall"
            writer.stop()
            store.close()
    finally:
        pi.close()
    print("  decide modes (vector/llm/hybrid) ok")


def test_retro_relaxed_gate() -> None:
    """Retrospective questions run the hybrid prefilter 0.10 looser (board
    0.75 normal / 0.65 retro)."""
    pi = FakePi()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            s, store, writer, sess = _make_session(
                tmp, MEMORY_DECISION_MODE="hybrid", MEMORY_PI_URL=pi.url)
            sess.note_user_turn("我刚买了一台尼康 FM2 胶片相机，很喜欢")
            writer.drain()
            retro_q = "你还记得我之前说的那台胶片相机是什么型号"
            normal_q = "胶片相机是什么型号"
            base = 0.75
            _set(s, "memory_retrieval_prefilter_score", base)
            assert abs(sess._prefilter_gate(retro_q) - (base - 0.10)) < 1e-9
            assert abs(sess._prefilter_gate(normal_q) - base) < 1e-9
            # probe the real raw score, then pin the prefilter JUST above it:
            # only the retro relaxation lets this query reach /decide
            search_q, _ = sess._prepare_query(retro_q)
            found = sess.retriever.search(sess.conversation_id, search_q, limit=4)
            raw = max((c.raw for c in found), default=0.0)
            assert 0.05 < raw < 0.85, f"probe assumption broke: raw={raw}"
            _set(s, "memory_retrieval_prefilter_score", raw + 0.05)
            before = len(pi.httpd.decide_requests)
            got = sess.recall_for_turn(retro_q)
            assert got, "retro question must pass the relaxed prefilter"
            assert len(pi.httpd.decide_requests) == before + 1
            writer.stop()
            store.close()
    finally:
        pi.close()
    print("  retro relaxed gate ok")


def test_unreachable_degrades_to_vector() -> None:
    """pi down (TCP refused): hybrid and llm both fall back to the local-vector
    behavior — recall works and the turn is not blocked."""
    with tempfile.TemporaryDirectory() as tmp:
        s, store, writer, sess = _make_session(
            tmp, MEMORY_DECISION_MODE="hybrid", MEMORY_PI_URL="http://127.0.0.1:9",
            MEMORY_PI_DECIDE_TIMEOUT_S=1)
        sess.note_user_turn("我刚买了一台尼康 FM2 胶片相机，很喜欢")
        writer.drain()
        t0 = time.monotonic()
        got = sess.recall_for_turn("我刚才说的那台胶片相机是什么型号")
        assert got, "unreachable pi must degrade hybrid to vector recall"
        assert time.monotonic() - t0 < 5.0, "degradation must not stall the turn"
        writer.stop()
        store.close()
    with tempfile.TemporaryDirectory() as tmp:
        s, store, writer, sess = _make_session(
            tmp, MEMORY_DECISION_MODE="llm", MEMORY_PI_URL="http://127.0.0.1:9",
            MEMORY_PI_DECIDE_TIMEOUT_S=1)
        sess.note_user_turn("我刚买了一台尼康 FM2 胶片相机，很喜欢")
        writer.drain()
        got = sess.recall_for_turn("我刚才说的那台胶片相机是什么型号")
        assert got, "unreachable pi must degrade llm to vector recall"
        writer.stop()
        store.close()
    print("  unreachable pi degrades to vector ok")


def test_compact_provider_pi() -> None:
    """provider "pi": /compact summary joins the summary layer, pins the pinned
    layer; empty summary and 5xx degrade to verbatim-tail-only without error."""
    pi = FakePi()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            s = _settings(tmp, MEMORY_SUMMARY_PROVIDER="pi", MEMORY_PI_URL=pi.url,
                          MEMORY_DECISION_MODE="vector")
            store = MemoryStore(s)
            store.open()
            for i, (u, a) in enumerate([("我买了一台尼康相机", "很棒的相机。"),
                                        ("周末去杭州拍照", "记得带胶卷。")]):
                store.add_item("c1", KIND_UTTERANCE, text=u, role="user", session_ts=i * 10.0)
                store.add_item("c1", KIND_UTTERANCE, text=a, role="assistant",
                               session_ts=i * 10.0 + 5.0)
            ro = RolloverManager(s, store, "c1", base_system_prompt="BASE_PROMPT",
                                 lang_getter=lambda: "zh")
            messages, kept, est = asyncio.run(ro.build_prefix())
            body = messages[0]["content"]
            assert "用户买了一台相机,预算两千元。" in body, "pi summary must join the summary layer"
            assert "用户预算 2000 元" in body, "pi pins must join the pinned layer"
            assert "置顶记忆" in body
            assert body.index("用户预算 2000 元") < body.index("用户买了一台相机,预算两千元。"), \
                "pins sit ahead of the summary"
            assert "记忆库覆盖" in body and messages[1:], "tail must still be there"
            assert len(pi.httpd.compact_requests) == 1
            req = pi.httpd.compact_requests[0]
            assert req["conversation_id"] == "c1"
            assert "尼康相机" in req["journal"] and "杭州" in req["journal"]

            # empty summary → verbatim-tail-only, no error
            pi.httpd.compact_response = {"summary": "", "pins": []}
            body = asyncio.run(ro.build_prefix())[0][0]["content"]
            assert "用户买了一台相机" not in body and "记忆库覆盖" in body

            # 5xx → same degrade (the client retries 3x, then None)
            pi.httpd.compact_response = None
            body = asyncio.run(ro.build_prefix())[0][0]["content"]
            assert "记忆库覆盖" in body
            store.close()
    finally:
        pi.close()
    print("  compact provider=pi ok")


def test_compact_prefetch() -> None:
    pi = FakePi()
    try:
        # ---- hit: a ready prefetch is consumed, /compact runs exactly once ----
        with tempfile.TemporaryDirectory() as tmp:
            s = _settings(tmp, MEMORY_SUMMARY_PROVIDER="pi", MEMORY_PI_URL=pi.url,
                          MEMORY_DECISION_MODE="vector", MEMORY_ROLLOVER_IDLE_TOKENS=100)
            store = MemoryStore(s)
            store.open()
            store.add_item("c1", KIND_UTTERANCE, text="我买了一台相机", role="user", session_ts=1.0)
            ro = RolloverManager(s, store, "c1", lang_getter=lambda: "zh")
            assert not ro.maybe_prefetch_compact(50), "below idle*0.6=60: no prefetch"
            before = len(pi.httpd.compact_requests)
            assert ro.maybe_prefetch_compact(80), "past the floor: prefetch starts"
            assert not ro.maybe_prefetch_compact(90), "one in-flight prefetch only"
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline:
                pf = ro._prefetch
                if pf is not None and pf.status != "running":
                    break
                time.sleep(0.02)
            assert ro._prefetch is not None and ro._prefetch.status == "ready"
            assert len(pi.httpd.compact_requests) == before + 1
            body = asyncio.run(ro.build_prefix())[0][0]["content"]
            assert "用户买了一台相机,预算两千元。" in body
            assert len(pi.httpd.compact_requests) == before + 1, \
                "build_prefix must consume the cache, not re-run /compact"
            assert ro._prefetch is None, "a consumed prefetch is popped"
            store.close()

        # ---- in-flight: build_prefix joins the running prefetch (≤30s cap) ----
        with tempfile.TemporaryDirectory() as tmp:
            slow = FakePi(compact_delay=0.8)
            try:
                s = _settings(tmp, MEMORY_SUMMARY_PROVIDER="pi", MEMORY_PI_URL=slow.url,
                              MEMORY_DECISION_MODE="vector", MEMORY_ROLLOVER_IDLE_TOKENS=100)
                store = MemoryStore(s)
                store.open()
                store.add_item("c1", KIND_UTTERANCE, text="我买了一台相机", role="user",
                               session_ts=1.0)
                ro = RolloverManager(s, store, "c1", lang_getter=lambda: "zh")
                assert ro.maybe_prefetch_compact(80)
                t0 = time.monotonic()
                body = asyncio.run(ro.build_prefix())[0][0]["content"]
                elapsed = time.monotonic() - t0
                assert "用户买了一台相机,预算两千元。" in body
                assert elapsed >= 0.5, f"build_prefix must join the in-flight prefetch ({elapsed})"
                assert len(slow.httpd.compact_requests) == 1, "join, not a duplicate /compact"
            finally:
                slow.close()
            store.close()

        # ---- failure: prefetch error degrades to the synchronous path ----
        with tempfile.TemporaryDirectory() as tmp:
            s = _settings(tmp, MEMORY_SUMMARY_PROVIDER="pi",
                          MEMORY_PI_URL="http://127.0.0.1:9",
                          MEMORY_DECISION_MODE="vector", MEMORY_ROLLOVER_IDLE_TOKENS=100,
                          MEMORY_PI_COMPACT_TIMEOUT_S=1)
            store = MemoryStore(s)
            store.open()
            store.add_item("c1", KIND_UTTERANCE, text="我买了一台相机", role="user", session_ts=1.0)
            ro = RolloverManager(s, store, "c1", lang_getter=lambda: "zh")
            assert ro.maybe_prefetch_compact(80)
            deadline = time.monotonic() + 15.0
            while time.monotonic() < deadline:
                pf = ro._prefetch
                if pf is not None and pf.status != "running":
                    break
                time.sleep(0.02)
            assert ro._prefetch is not None and ro._prefetch.status == "error"
            messages, kept, est = asyncio.run(ro.build_prefix())
            assert messages and messages[1:], "verbatim tail must survive a failed prefetch"
            store.close()

        # ---- non-pi providers never prefetch ----
        with tempfile.TemporaryDirectory() as tmp:
            s = _settings(tmp, MEMORY_SUMMARY_PROVIDER="none",
                          MEMORY_ROLLOVER_IDLE_TOKENS=100)
            store = MemoryStore(s)
            store.open()
            ro = RolloverManager(s, store, "c1")
            assert not ro.maybe_prefetch_compact(99999)
            store.close()
    finally:
        pi.close()
    print("  compact prefetch (hit/join/degrade) ok")


def test_writer_utterance_dedup() -> None:
    """Same (role, whitespace-normalized text) is stored once per conversation;
    pinned items never pass through the writer and are unaffected."""
    with tempfile.TemporaryDirectory() as tmp:
        s, store, writer, sess = _make_session(tmp, MEMORY_DECISION_MODE="vector")
        sess.note_user_turn("你好 世界")
        sess.note_user_turn("你好  世界")   # whitespace variant: duplicate
        sess.note_user_turn("你好 世界")   # exact duplicate
        sess.note_assistant_turn("我很好")
        sess.note_assistant_turn("我很好")
        sess.note_user_turn("今天天气怎么样")  # distinct: stored
        writer.drain()
        items = store.recent("c1", [KIND_UTTERANCE], limit=20)
        texts = sorted((i.role, " ".join(i.text.split())) for i in items)
        assert texts == [("assistant", "我很好"), ("user", "今天天气怎么样"),
                         ("user", "你好 世界")], texts
        # pinned rows are written via the store directly — dedup never applies
        store.add_item("c1", KIND_PINNED, text="用户是左撇子", role="user")
        store.add_item("c1", KIND_PINNED, text="用户是左撇子", role="user")
        assert store.count("c1") == 5
        # forget() resets the seen set: the same text may be stored again
        writer.forget("c1")
        sess.note_user_turn("你好 世界")
        writer.drain()
        assert store.count("c1") == 6
        writer.stop()
        store.close()
    print("  writer utterance dedup ok")


def test_interrupted_commit_false() -> None:
    """commit=False: the truncated turn never lands in the store but rides the
    compact journal; note_rollover starts the buffer fresh."""
    pi = FakePi()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            s, store, writer, sess = _make_session(
                tmp, MEMORY_DECISION_MODE="vector", MEMORY_SUMMARY_PROVIDER="pi",
                MEMORY_PI_URL=pi.url)
            sess.note_user_turn("给我讲讲这台相机")
            sess.note_assistant_turn("这是一台尼康 FM2,它", commit=False)  # barged-in
            sess.note_user_turn("算了,说说杭州吧")
            sess.note_assistant_turn("杭州适合拍照。")
            writer.drain()
            stored = [i.text for i in store.recent("c1", [KIND_UTTERANCE], limit=20)]
            assert not any("这是一台尼康" in t for t in stored), \
                "an interrupted turn must never be stored"
            assert "杭州适合拍照。" in stored and len(stored) == 3
            assert sess.uncommitted_turns() == [("assistant", "这是一台尼康 FM2,它")]
            est_after = sess._est_tokens
            assert est_after > 0, "the interrupted text still occupies KV"

            ro = RolloverManager(s, store, "c1", lang_getter=lambda: "zh",
                                 journal_extra=sess.uncommitted_turns)
            asyncio.run(ro.build_prefix())
            assert pi.httpd.compact_requests, "pi provider must call /compact"
            journal = pi.httpd.compact_requests[-1]["journal"]
            assert "这是一台尼康 FM2,它" in journal and "[interrupted]" in journal, \
                "uncommitted turns are compact-journal context"
            assert "杭州适合拍照。" in journal

            sess.note_rollover([], text_tokens=10.0)
            assert sess.uncommitted_turns() == [], "rollover starts the buffer fresh"
            writer.stop()
            store.close()
    finally:
        pi.close()
    print("  interrupted commit=False ok")


def main() -> None:
    test_decide_modes()
    test_retro_relaxed_gate()
    test_unreachable_degrades_to_vector()
    test_compact_provider_pi()
    test_compact_prefetch()
    test_writer_utterance_dedup()
    test_interrupted_commit_false()
    print("memory-pi: all checks passed")


if __name__ == "__main__":
    main()
