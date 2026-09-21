"""History-query routing and temporal regression tests; no models or network."""
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from .test_memory_pi import _make_session
from ..memory.retrieval import Candidate
from ..memory.rewrite import parse_time_window
from ..memory.pi_client import PiAgentClient


class RecallRegression(unittest.TestCase):
    def test_selection_client_rejects_unknown_ids(self):
        settings = SimpleNamespace(memory_pi_url="http://127.0.0.1:1")
        client = PiAgentClient(settings)
        for payload in ({"ids": [2]}, {"ids": [True]}, {"ids": "1"}, None):
            with patch("server.memory.pi_client._post_json", return_value=payload):
                self.assertIsNone(client.select("q", [{"id": 1, "text": "fact"}], timeout_s=1))
        with patch("server.memory.pi_client._post_json", return_value={"ids": []}):
            self.assertEqual(client.select("q", [{"id": 1, "text": "fact"}], timeout_s=1), [])

    def test_earliest_is_not_first_minute(self):
        self.assertIsNone(parse_time_window("最早我告诉你的鼠标的牌子是什么", 610))
        self.assertEqual(parse_time_window("开头的画面是什么", 610), (0, 60))
        self.assertEqual(parse_time_window("看 t=1:15", 610), (45, 105))

    def test_explicit_history_rewrite_keeps_evidence_gates(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings, store, writer, session = _make_session(
                tmp, MEMORY_DECISION_MODE="hybrid", MEMORY_RETRIEVAL_PREFILTER_SCORE="0.75")
            try:
                session.session_ts = lambda: 75.0
                session.note_assistant_turn("鼠标上有 ATK 标志")
                writer.drain()
                item = store.recent("c1", limit=1)[0]
                calls = []
                response = {"retrieve": True, "query": "鼠标品牌", "reason": "history"}
                session._pi = SimpleNamespace(reachable=lambda: True, decide=lambda *a: response,
                                              select=lambda *a, **kw: [item.id])

                def candidates(query):
                    calls.append(query)
                    return [Candidate(item, relevance=1, raw=0.7 if query == "鼠标品牌" else 0.468, late=True)]

                session._candidates = candidates
                session.retriever.diversify = lambda c, q, limit: c
                query = "最早我告诉你的鼠标的牌子是什么"
                result = session.recall_for_turn(query)
                self.assertTrue(result)
                self.assertIn("ATK", result.block)
                self.assertEqual(calls, ["鼠标品牌"])
                session._pi.select = lambda *a, **kw: []
                self.assertFalse(session.recall_for_turn(query), "no evidence must not be injected")
                session._pi.select = lambda *a, **kw: None
                self.assertFalse(session.recall_for_turn(query), "old sidecar keeps conservative gate")
                session._pi.select = lambda *a, **kw: [item.id]
                session.mark_injected(result.ids)
                self.assertFalse(session.recall_for_turn(query), "context de-dup must remain")
                response["retrieve"] = False
                calls.clear()
                self.assertFalse(session.recall_for_turn(query))
                self.assertEqual(calls, [], "decision veto must precede search")
                response.update(retrieve=True, query="unrelated")
                session._injected.clear()
                self.assertTrue(session.recall_for_turn(query), "temporal lookup judges the original question independently of semantic Top-K")
            finally:
                writer.stop()
                store.close()


if __name__ == "__main__":
    unittest.main()
