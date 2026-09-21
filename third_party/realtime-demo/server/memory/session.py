"""Per-session memory facade — the only object the orchestrator talks to.

Everything session-scoped lives here and dies with the session: the injected-id
table, the prefetch cache, the rolling conversation language, the keyframe
throttle. The shared, process-wide pieces (store, writer, embedders) are passed
in. Retrieval is always scoped to this session's `conversation_id`, so one
session can never read another's memories.

Two-lane read path: `prefetch()` runs off the ASR partial thread and parks
ranked candidates in a cache (gold evidence is usually retrievable well before
the user stops talking); `recall_for_turn()` then only has to re-gate against
the final transcript, so the turn itself pays almost nothing.

Re-injection (design §5): injected items STAY in retrieval and ranking —
excluding them up front made an explicit re-request ("不是那个,我是说之前那台相机")
silently unanswerable. They are dropped only AFTER ranking, by the distance
gate (fewer than `memory_reinject_distance` text-KV tokens since last shown, or
`memory_reinject_max_copies` copies already injected). Positions are the
worker's exact text-token count when the orchestrator plumbs it through, else
an internal running estimate of noted turns + injected blocks.
"""
from __future__ import annotations
import hashlib
from collections import OrderedDict

import asyncio
import threading
import time
from dataclasses import dataclass, field, replace
from itertools import islice
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from ..config import Settings
from ..logging_conf import get_logger
from . import inject as inject_mod
from . import lang as lang_mod
from . import rewrite as rewrite_mod
from .pi_client import PiAgentClient
from .lifecycle import SessionLifetime, while_open
from .retrieval import Candidate, Retriever, has_past_reference
from .store import KIND_FRAME, KIND_UTTERANCE, MemoryStore
from .writer import MemoryWriter

log = get_logger(__name__)

# Retrospective question markers (board parity): loosen the hybrid prefilter
# for questions that clearly refer backwards ("刚才说的…", "还记得那台…").
_RETRO_QUESTION_PATTERNS = (
    "刚才",
    "之前",
    "是不是",
    "有没有",
    "记得",
    "还记得",
    "那会儿",
    "那台",
    "那个",
    "那时候",
)
# board runs its retro prefilter 0.10 below the normal one (0.75 vs 0.65)
_RETRO_PREFILTER_RELAX = 0.10

_RECENT_TURNS_FOR_DECIDE = 8


def is_retro_question(text: str) -> bool:
    return any(pattern in (text or "") for pattern in _RETRO_QUESTION_PATTERNS)


@dataclass
class RecallResult:
    block: str
    items: List[Dict[str, Any]]
    ids: List[int]
    # JPEGs to re-push through the frame queue (channel V). Empty unless
    # MEMORY_INJECT_FRAMES — a matched frame with no caption is worth nothing as
    # text, so either the pixels come back or the frame is not recalled at all.
    frames: List[bytes] = field(default_factory=list)
    reason: Optional[str] = None

    def __bool__(self) -> bool:
        return bool(self.block)


EMPTY_RECALL = RecallResult(block="", items=[], ids=[], frames=[])


@dataclass
class _InjectedEntry:
    """Per-item re-injection bookkeeping, in TEXT-KV TOKEN positions."""
    copies: int = 0          # times injected this session (cap: memory_reinject_max_copies)
    last_tokens: float = 0.0  # session text-token position when last shown
    full: bool = True
    covered: Set[bytes] = field(default_factory=set)


class MemorySession:
    def __init__(self, conversation_id: str, settings: Settings, store: MemoryStore,
                 writer: MemoryWriter, *, default_lang: str = "zh", facts: Any = None) -> None:
        self.conversation_id = conversation_id
        self.settings = settings
        self.lifetime = SessionLifetime()
        self._close_lock = threading.Lock()
        self._cleaned = False
        self.store = store
        self.writer = writer
        self.retriever = Retriever(store, writer.text, writer.image, settings)
        # background fact extraction (memory/facts.py); None = facts off
        self._facts = facts
        self.lang_state = lang_mod.LanguageState(default=default_lang)
        self._t0 = time.monotonic()
        self._injected: Dict[int, _InjectedEntry] = {}
        self._native_context = OrderedDict()
        self._suppressed: Set[int] = set()
        self._last_frame_at = 0.0
        self._prefetch: Optional[Tuple[str, List[Candidate]]] = None
        self._lock = threading.Lock()
        # pi_agent sidecar client (decide gate + compact provider); the decide
        # gate only engages for memory_decision_mode llm/hybrid AND a reachable
        # pi — everything else degrades to the local-vector behavior
        self._pi = PiAgentClient(settings)
        # interrupted assistant turns (commit=False): they occupied KV and are
        # compact-journal context, but a truncated answer never lands in the
        # long-term store. Drained by the rollover journal via uncommitted_turns.
        self._uncommitted: List[Tuple[str, str]] = []
        # internal text-KV estimate, used only when the orchestrator does not
        # pass the worker's exact count (keeps the distance gate testable and
        # self-contained): noted turns + injected blocks, in estimate_tokens
        self._est_tokens = 0.0
        # estimated INJECTED tokens over the session's life. Deliberately
        # survives note_rollover: exhaustion is a rollover signal (design §5),
        # not a per-KV budget
        self._lifetime_tokens = 0.0
        # estimate of the last formatted block, committed by mark_injected
        self._pending_block_tokens = 0
        self._pending_injection_bodies = {}
        self._distance_reason = None
        self.stats = {"recalls": 0, "injected": 0, "gated_out": 0}

    # ---- clock ----

    def session_ts(self, explicit: Optional[float] = None) -> float:
        if explicit is not None:
            return float(explicit)
        return time.monotonic() - self._t0

    @property
    def language(self) -> str:
        return self.lang_state.current

    # ---- write path (all non-blocking) ----

    @while_open()
    def note_user_turn(self, text: str, *, media_ts: Optional[float] = None,
                       session_ts: Optional[float] = None) -> None:
        text = (text or "").strip()
        if not text:
            return
        self._est_tokens += inject_mod.estimate_tokens(text)
        self.lang_state.observe(text)
        self.writer.note_utterance(
            self.conversation_id, "user", text,
            lang=lang_mod.detect_lang(text, default=self.language),
            session_ts=self.session_ts(session_ts), media_ts=media_ts, importance=0.65,
            lifetime=self.lifetime)

    @while_open()
    def note_assistant_turn(self, text: str, *, media_ts: Optional[float] = None,
                            commit: bool = True, historical_answer: bool = False) -> None:
        text = (text or "").strip()
        if not text:
            return
        self._est_tokens += inject_mod.estimate_tokens(text)
        if not commit:
            # barge-in / <|eot_id|> truncated turn: the text DID occupy KV and
            # stays available as compact-journal context (uncommitted_turns),
            # but a half-finished answer must never enter long-term memory
            with self._lock:
                if sum(len(t.encode()) for _, t in self._uncommitted) + len(text.encode()) <= max(1, self.settings.memory_uncommitted_bytes):
                    self._uncommitted.append(("assistant", text))
                else:
                    self.stats['uncommitted_rejected'] = self.stats.get('uncommitted_rejected', 0) + 1
                    log.warning('memory interrupted-history buffer full session=%s', self.conversation_id)
            return
        self.writer.note_utterance(
            self.conversation_id, "assistant", text,
            lang=lang_mod.detect_lang(text, default=self.language),
            session_ts=self.session_ts(), media_ts=media_ts, importance=0.4,
            lifetime=self.lifetime, historical_answer=historical_answer)

    def uncommitted_turns(self) -> List[Tuple[str, str]]:
        """(role, text) interrupted turns kept as compact context only."""
        with self._lock:
            return list(self._uncommitted)

    @while_open()
    def note_frame(self, jpeg: bytes, timestamp: Optional[float] = None,
                   media_ts: Optional[float] = None) -> None:
        """Hot-path throttle only — real dedup happens on the writer thread."""
        now = time.monotonic()
        if (now - self._last_frame_at) < max(0.0, float(self.settings.memory_keyframe_min_interval_s)):
            return
        self._last_frame_at = now
        self.writer.note_frame(self.conversation_id, jpeg, session_ts=self.session_ts(timestamp),
                               media_ts=media_ts, lang=self.language, lifetime=self.lifetime)

    # ---- read path ----

    @while_open()
    def prefetch(self, partial_text: str) -> None:
        """Best-effort warm-up from an ASR partial. Never raises, never gates."""
        text = (partial_text or "").strip()
        if len(text) < 2 or not self.settings.memory_prefetch_on_partials:
            return
        try:
            # the same rewrite rules as recall_for_turn, so the cache key and
            # the eventual query agree (the time WINDOW is applied at recall)
            search_q, _ = self._prepare_query(text)
            found = self.retriever.search(self.conversation_id, search_q, exclude=self._skip_ids(),
                                          limit=max(1, int(self.settings.memory_retrieval_topk)))
            if found:
                with self._lock:
                    self._prefetch = (search_q, self.retriever.rank(found))
        except Exception as exc:  # noqa: BLE001
            log.debug("memory prefetch failed: %s", exc)

    def _skip_ids(self) -> Set[int]:
        # ONLY suppression excludes up front. Injected ids stay in retrieval and
        # ranking; the distance gate decides after ranking (design §5).
        return set(self._suppressed)

    def _prepare_query(self, text: str) -> Tuple[str, Optional[rewrite_mod.TimeWindow]]:
        """Rule-based pre-retrieval rewrite (memory/rewrite.py): a confident
        relative-time parse yields a session-time window; a short deictic
        past-reference query is augmented with the most recent user turns so
        raw embedding similarity has something to bite on. Conservative:
        anything unrecognized passes through untouched."""
        window = rewrite_mod.parse_time_window(text, self.session_ts())
        query = text
        if rewrite_mod.is_deictic_query(text):
            recent = self.store.recent(self.conversation_id, [KIND_UTTERANCE], limit=8)
            texts = [i.text for i in recent if i.role == "user" and i.text][:2]
            query = rewrite_mod.augment_query(text, texts)
        return query, window

    # ---- pi_agent decide gate (memory_decision_mode llm/hybrid) ----

    def _recent_turns_text(self, limit: int = _RECENT_TURNS_FOR_DECIDE) -> str:
        """Chronological "role: text" lines from the store — the /decide context."""
        try:
            recent = self.store.recent(self.conversation_id, [KIND_UTTERANCE], limit=limit)
        except Exception:  # noqa: BLE001
            return ""
        return "\n".join(f"{item.role or 'user'}: {item.text}"
                         for item in reversed(recent) if item.text)

    def _pi_decide(self, query: str) -> Optional[Dict[str, Any]]:
        """Ask pi_agent /decide; None = unreachable/failed → caller degrades to
        the local-vector behavior, never blocking the turn."""
        pi = self._pi
        if pi is None or not pi.reachable():
            return None
        return pi.decide(self.conversation_id, self._recent_turns_text(), query)

    def _prefilter_gate(self, query: str) -> float:
        """Hybrid stage-1 threshold; retrospective questions run looser (board
        0.75 normal / 0.65 retro)."""
        gate = float(self.settings.memory_retrieval_prefilter_score)
        if is_retro_question(query):
            gate -= _RETRO_PREFILTER_RELAX
        return gate

    def _hybrid_decide(self, query: str, candidates: List[Candidate]) -> bool:
        """Stage 1: the best ABSOLUTE raw score must clear the prefilter before
        pi is consulted at all. Stage 2: pi /decide confirms. Pi unreachable or
        a failed decide both degrade to pure vector behavior (proceed)."""
        pi = self._pi
        if pi is None or not pi.reachable():
            return True
        if max((c.raw for c in candidates), default=0.0) < self._prefilter_gate(query):
            log.info("memory recall session=%s stage=prefilter rejected threshold=%.3f best=%.3f",
                     self.conversation_id, self._prefilter_gate(query),
                     max((c.raw for c in candidates), default=0.0))
            return False
        decision = pi.decide(self.conversation_id, self._recent_turns_text(), query)
        if decision is None:
            return True
        return bool(decision.get("retrieve"))

    @while_open(EMPTY_RECALL)
    def recall_for_turn(self, text: str, *, now_tokens: Optional[float] = None) -> RecallResult:
        """Blocking (call via to_thread): rewrite → search → rank → time window
        → re-injection distance gate → diversify → admission gate → format.

        `now_tokens` is the session's text-KV token count; when None the
        internal running estimate is used (see module docstring)."""
        query = (text or "").strip()
        if not query:
            return EMPTY_RECALL
        if self._lifetime_tokens >= max(1, int(self.settings.memory_inject_session_max_tokens)):
            # lifetime budget exhausted: stop recalling SILENTLY — this is a
            # rollover signal, not a reason to raise the cap (design §5)
            log.info("memory recall session=%s stage=lifetime_budget exhausted", self.conversation_id)
            return EMPTY_RECALL
        if now_tokens is not None:
            # the worker's exact count supersedes the estimate, never rewinds it
            self._est_tokens = max(self._est_tokens, float(now_tokens))
        self._pending_block_tokens = 0
        try:
            self._pending_injection_bodies.clear()
            decision_started = time.monotonic()
            search_q, window = self._prepare_query(query)
            mode = (self.settings.memory_decision_mode or "vector").strip().lower()
            # Explicit history requests need query resolution BEFORE score gating.
            # Keep final evidence admission and KV de-duplication below unchanged.
            decision_first = mode == "llm" or (mode == "hybrid" and
                (has_past_reference(query) or rewrite_mod.temporal_order(query) is not None))
            if decision_first:
                # pi /decide gates the whole turn; a retrieve=false verdict
                # skips recall, a query rewrite overrides the search query
                decision = self._pi_decide(query)
                if decision is not None:
                    if not decision.get("retrieve"):
                        log.info("memory recall session=%s stage=decision rejected", self.conversation_id)
                        self.stats["gated_out"] += 1
                        return EMPTY_RECALL
                    override = str(decision.get("query") or "").strip()
                    if override:
                        search_q = override
            candidates = self._candidates(search_q)
            log.info("memory recall session=%s stage=search rewritten=%s candidates=%s",
                     self.conversation_id, search_q != query,
                     [(c.item.id, c.space, round(c.raw, 3)) for c in candidates])
            log.info('memory recall session=%s stage=candidate_meta items=%s', self.conversation_id,
                     [(c.item.id, c.item.session_ts, c.item.kind, len(c.item.text)) for c in candidates])
            order = rewrite_mod.temporal_order(query)
            if order and mode in ('hybrid', 'llm'):
                return self._recall_temporal(query, order, decision_started)
            if not candidates:
                return EMPTY_RECALL
            if mode == "hybrid" and not decision_first and not self._hybrid_decide(query, candidates):
                self.stats["gated_out"] += 1
                return EMPTY_RECALL
            candidates = self._time_gate(candidates, window)
            log.info("memory recall session=%s stage=time window=%s ids=%s",
                     self.conversation_id, window, [c.item.id for c in candidates])
            if not candidates:
                self.stats["gated_out"] += 1
                return EMPTY_RECALL
            limit = max(1, int(self.settings.memory_retrieval_topk))
            # diversify/gate read the RAW query — the augmentation is search-only
            diverse = self.retriever.diversify(candidates, query, limit=limit)
            keep = self.retriever.gate(query, diverse)
            log.info('memory recall session=%s stage=local_admission diverse=%s accepted=%s',
                     self.conversation_id, [c.item.id for c in diverse], [c.item.id for c in keep])
            verified_empty = False
            if mode == 'hybrid':
                # Already-present context is evidence too; it must be examined
                # before deciding that a textual history question has no answer.
                contextual = self.store.recent(self.conversation_id, [KIND_UTTERANCE], limit=8)
                contextual += list(self.store.get_items(list(self._injected)).values())
                seen = {c.item.id for c in keep}
                for item in contextual:
                    if len(keep) >= 16:
                        break
                    if item.conversation_id != self.conversation_id:
                        continue
                    if item.id not in seen and (self._context_key(item.role, item.text) in self._native_context or item.id in self._injected):
                        keep.append(Candidate(item, relevance=0.0))
                        seen.add(item.id)
            if mode == "hybrid" and keep and self._pi is not None and self._pi.reachable():
                remaining = float(self.settings.memory_pi_decide_timeout_s) - (time.monotonic() - decision_started)
                selected = self._pi.select(query, [
                    {"id": c.item.id, "role": c.item.role, "session_ts": c.item.session_ts,
                     "text": (c.item.text or "")[:2000]}
                    for c in keep], timeout_s=remaining, recent_turns=self._recent_turns_text())
                if selected is None:
                    # Old/unavailable sidecars retain the original conservative prefilter.
                    keep = [c for c in keep if c.raw >= self._prefilter_gate(query)]
                else:
                    verified_empty = not selected
                    keep = [c for c in keep if c.item.id in selected]
                log.info("memory recall session=%s stage=evidence available=%s ids=%s",
                         self.conversation_id, selected is not None, [c.item.id for c in keep])
            # Select the right evidence before suppressing copies already in KV.
            # Removing a retained correction first can promote an obsolete fact.
            keep = self._distance_gate(keep, self._est_tokens)
            log.info("memory recall session=%s stage=context ids=%s",
                     self.conversation_id, [c.item.id for c in keep])
            log.info("memory recall session=%s stage=admission diverse=%s kept=%s",
                     self.conversation_id, [c.item.id for c in diverse], [c.item.id for c in keep])
            self.stats["recalls"] += 1
            if not keep:
                self.stats["gated_out"] += 1
                return RecallResult('', [], [], reason='candidate_miss' if verified_empty else self._distance_reason)
            result = self._format(keep)
            log.info("memory recall session=%s stage=format ids=%s", self.conversation_id, result.ids)
            return result
        except Exception as exc:  # noqa: BLE001 — memory never breaks a turn
            log.warning("memory recall failed: %s", exc)
            return EMPTY_RECALL

    def _recall_temporal(self, query, order, started):
        items, complete = self.store.timeline(self.conversation_id, latest=order == 'latest',
            limit=max(1, self.settings.memory_temporal_scan_items),
            max_bytes=max(1, self.settings.memory_temporal_scan_bytes))
        log.info('memory recall session=%s stage=temporal order=%s rows=%d complete=%s',
                 self.conversation_id, order, len(items), complete)
        if self._pi is None or not self._pi.reachable():
            return RecallResult('', [], [], reason='selector_unavailable')
        def spans():
            serial = 1
            for item in items:
                if item.id in self._suppressed:
                    continue
                chunks = inject_mod.verbatim_chunks(item.text, self.settings.memory_inject_max_tokens-12)
                if order == 'latest':
                    chunks.reverse()
                for text in chunks:
                    candidate_id = serial
                    serial += 1
                    yield candidate_id, replace(item, text=text)
        pending = spans()
        offset = 0
        while True:
            batch = list(islice(pending, 16))
            if not batch:
                break
            remaining = self.settings.memory_pi_decide_timeout_s - (time.monotonic() - started)
            if remaining <= 0:
                return RecallResult('', [], [], reason='temporal_budget')
            selected = self._pi.select(query, [dict(id=key, role=i.role, session_ts=i.session_ts,
                text=i.text) for key, i in batch], timeout_s=remaining, recent_turns='')
            log.info('memory recall session=%s stage=temporal_evidence offset=%d ids=%s selected=%s',
                     self.conversation_id, offset, [(key,i.id,len(i.text)) for key,i in batch], selected)
            if selected is None:
                return RecallResult('', [], [], reason='selector_failed')
            # Stop at the first related record in the requested temporal order,
            # before KV de-duplication can promote a later/earlier occurrence.
            chosen = next((i for key,i in batch if key in selected), None)
            if chosen is not None:
                keep = self._distance_gate([Candidate(chosen, relevance=1.0)], self._est_tokens)
                self.stats['recalls'] += 1
                if not keep:
                    return RecallResult('', [], [], reason=self._distance_reason or 'context_present')
                return self._format(keep, shorten=False)
            offset += len(batch)
        return RecallResult('', [], [], reason='candidate_miss' if complete else 'temporal_budget')

    def _candidates(self, query: str) -> List[Candidate]:
        with self._lock:
            cached = self._prefetch
            self._prefetch = None
        if cached is not None:
            cached_q, cands = cached
            # the partial is a prefix of the final transcript often enough that
            # reusing it is the common case; otherwise fall through to a search
            if cached_q and (query.startswith(cached_q) or cached_q.startswith(query)):
                skip = self._skip_ids()
                fresh = [c for c in cands if c.item.id not in skip]
                if fresh:
                    return fresh
        found = self.retriever.search(self.conversation_id, query, exclude=self._skip_ids(),
                                      limit=max(1, int(self.settings.memory_retrieval_topk)))
        return self.retriever.rank(found) if found else []

    def _time_gate(self, candidates: List[Candidate],
                   window: Optional[rewrite_mod.TimeWindow]) -> List[Candidate]:
        """A confident relative-time window restricts candidates to items whose
        ORIGINAL session_ts falls inside it (boosted so they keep their rank
        edge). A window with no hits drops nothing — the parse was confident,
        the corpus may simply not reach back that far."""
        if window is None or not candidates:
            return candidates
        lo, hi = window
        inside = [c for c in candidates
                  if c.item.session_ts is not None and lo <= c.item.session_ts <= hi]
        if not inside:
            return candidates
        for cand in inside:
            cand.score += 0.5
        inside.sort(key=lambda c: c.score, reverse=True)
        return inside

    def _distance_gate(self, candidates: List[Candidate], now_tokens: float) -> List[Candidate]:
        """Re-injection control, applied AFTER ranking (design §5): an
        already-shown item competes with fresh ones and is dropped only if it
        is still inside the effective window — fewer than
        `memory_reinject_distance` text-KV tokens since it was last shown — or
        has hit its per-session copy cap. Fresh candidates are never touched."""
        max_copies = max(1, int(self.settings.memory_reinject_max_copies))
        distance = max(0, int(self.settings.memory_reinject_distance))
        out: List[Candidate] = []
        self._distance_reason = None
        originals = self.store.get_items([c.item.id for c in candidates])
        for cand in candidates:
            original = originals.get(cand.item.id, cand.item)
            if self._context_key(original.role, original.text) in self._native_context:
                self._distance_reason = 'context_present'
                continue
            entry = self._injected.get(cand.item.id)
            if entry is None:
                out.append(cand)
                continue
            if entry.copies >= max_copies:
                self._distance_reason = 'copy_budget'
                continue
            if not entry.full and self._body_key(cand.item.text) not in entry.covered:
                out.append(cand)
                continue
            if (now_tokens - entry.last_tokens) < distance:
                self._distance_reason = 'context_present'
                continue
            out.append(cand)
        return out

    def _format(self, keep: Sequence[Candidate], *, shorten=True) -> RecallResult:
        """Chronological by ORIGINAL time, under a hard token budget."""
        ordered = sorted(keep, key=lambda c: (c.item.session_ts or 0.0))
        budget = max(0, int(self.settings.memory_inject_max_tokens))
        budget = min(budget, max(0, int(self.settings.memory_inject_session_max_tokens-self._lifetime_tokens)))
        budget = max(0, budget - inject_mod.estimate_tokens(inject_mod.build_recall_block([''])))
        lines: List[str] = []
        items: List[Dict[str, Any]] = []
        ids: List[int] = []
        frames: List[bytes] = []
        used = 0
        originals = self.store.get_items([c.item.id for c in ordered])
        for cand in ordered:
            item = cand.item
            if item.conversation_id != self.conversation_id:
                continue
            is_frame = item.kind == KIND_FRAME
            jpeg = None
            if is_frame and not item.text:
                # cross-modal retrieval can match a frame that was never
                # captioned; it is only useful if the pixels come back with it
                if not self.settings.memory_inject_frames:
                    continue
                jpeg = self._load_frame(item.media_hash)
                if jpeg is None:
                    continue
            # re-injects ride the SHORT form (design §5): the first clause of
            # the verbatim text, not a byte-repeat of the whole turn
            entry = self._injected.get(item.id)
            if shorten and entry is not None and entry.full and entry.copies >= 1 and item.text:
                body = inject_mod.first_clause(item.text)
            else:
                body = item.text or ("画面" if self.language.startswith("zh") else "camera view")
            line = inject_mod.format_recall_line(body, item.session_ts, frame=is_frame)
            cost = inject_mod.estimate_tokens(line)
            if used + cost > budget:
                break
            lines.append(line)
            used += cost
            ids.append(item.id)
            self._pending_injection_bodies[item.id] = body
            if jpeg is not None:
                frames.append(jpeg)
            items.append({"id": item.id, "text": originals.get(item.id, item).text, "session_ts": item.session_ts,
                          "kind": item.kind, "media": item.media_hash, "score": round(cand.score, 4)})
        if not lines:
            return RecallResult('', [], [], reason='injection_budget')
        block = inject_mod.build_recall_block(lines)
        self._pending_block_tokens = inject_mod.estimate_tokens(block)
        return RecallResult(block=block, items=items, ids=ids, frames=frames)

    def _load_frame(self, media_hash: Optional[str]) -> Optional[bytes]:
        if media_hash and media_hash.startswith('memory:'):
            try:
                return self.store.frames.load(self.conversation_id, media_hash)
            except Exception:
                return None
        media = getattr(self.writer, "media", None)
        if not media_hash or media is None:
            return None
        try:
            if hasattr(media, 'load_bytes'):
                return media.load_bytes(media_hash)
            from ..persistence.media import normalize_hash
            digest = normalize_hash(media_hash)
            if digest is not None:
                with open(media.blob_path(digest), 'rb') as stream:
                    return stream.read()
            return None
        except Exception:  # noqa: BLE001
            return None

    @while_open()
    def mark_injected(self, ids: Sequence[int], *, text_tokens: Optional[float] = None,
                      block: Optional[str] = None) -> None:
        """Commit an injected block: bump each item's copy count, stamp the
        text-KV position it was shown at, and fold the block's estimated tokens
        into the lifetime budget and the position estimate.

        `text_tokens` is the worker's exact session text-token count when the
        orchestrator plumbs it through; otherwise the internal estimate."""
        pos = float(text_tokens) if text_tokens is not None else self._est_tokens
        owned = self.store.get_items(list(ids))
        ids = [i for i in ids if int(i) in owned and owned[int(i)].conversation_id == self.conversation_id]
        for item_id in ids:
            entry = self._injected.setdefault(int(item_id), _InjectedEntry(full=False))
            body = self._pending_injection_bodies.pop(item_id, None)
            entry.full = entry.full or body is None or self._body_key(body) == self._body_key(owned[int(item_id)].text)
            if body is not None:
                entry.covered.add(self._body_key(body))
            entry.copies += 1
            entry.last_tokens = pos
            try:
                self.store.mark_injected(int(item_id), self.conversation_id)
            except Exception:  # noqa: BLE001
                pass
        if ids:
            block_tokens = float(inject_mod.estimate_tokens(block) if block is not None else self._pending_block_tokens)
            self._pending_block_tokens = 0
            self._lifetime_tokens += block_tokens
            # the block itself occupies KV from here on — keep the estimate monotone
            self._est_tokens = max(self._est_tokens, pos) + block_tokens
            self.stats["injected"] += len(ids)

    @while_open()
    def note_context_turn(self, role: str, text: str) -> None:
        """Called only after model input acknowledgement or completed output."""
        if text and text.strip():
            key = self._context_key(role, text)
            self._native_context[key] = None
            self._native_context.move_to_end(key)
            while len(self._native_context) > max(1, self.settings.memory_session_max_items):
                self._native_context.popitem(last=False)

    @staticmethod
    def _context_key(role, text):
        return hashlib.sha256((str(role)+'\0'+(text or '').strip()).encode()).digest()

    @staticmethod
    def _body_key(text):
        return hashlib.sha256(inject_mod.sanitize_model_text(text).replace('\n', ' ').strip().encode()).digest()

    @while_open()
    def note_rollover(self, kept_item_ids: Sequence[int], *, text_tokens: float = 0.0) -> None:
        """Re-seat after a KV rollover (the rollover machinery itself is a
        separate workstream): the injected table is recomputed from EXACTLY the
        ids that went into the new prefix (design §5) — clearing it blindly
        would re-inject items already in the prefix as duplicates. Distances
        reset (each kept item was last shown at the new prefix's token count);
        `_suppressed` PERSISTS ("不是那个" survives a rollover); the lifetime
        injected-token budget does NOT reset.
        """
        self._injected = {int(i): _InjectedEntry(copies=1, last_tokens=float(text_tokens))
                          for i in kept_item_ids}
        self._native_context.clear()
        self._est_tokens = max(0.0, float(text_tokens))
        self._pending_block_tokens = 0
        with self._lock:
            # the interrupted-turn buffer was part of the compacted journal;
            # the new prefix already carries it — start fresh
            self._uncommitted.clear()

    @while_open()
    def suppress(self, ids: Sequence[int]) -> None:
        """User said 'not that one' — stop surfacing these for this session."""
        self._suppressed.update(int(i) for i in ids)

    # ---- background fact extraction (never on the hot path) ----

    @while_open()
    async def maybe_extract_facts(self, item_id: Optional[int], user_text: str,
                                  context_turns: Optional[Sequence[str]] = None) -> None:
        """Fire-and-forget from the orchestrator once an assistant reply
        finalizes a QA pair (a completed reply is the segment boundary,
        design §3). Facts enrich the RETRIEVAL INDEX KEY only — the model
        never sees them. Never raises; every failure is a silent no-op.
        """
        extractor = self._facts
        text = (user_text or "").strip()
        if extractor is None or not text:
            return
        if (self.settings.memory_fact_scope or "user") != "user":
            return
        try:
            if item_id is None:
                item_id = await self._resolve_user_item(text)
            if item_id is None or self.lifetime.closing:
                return
            if context_turns is None:
                context_turns = await asyncio.to_thread(self._recent_context, int(item_id))
            await extractor.extract_and_rekey(self.conversation_id, int(item_id), text,
                                              context_turns, lifetime=self.lifetime)
        except Exception as exc:  # noqa: BLE001 — memory must never kill a session
            log.debug("memory fact extraction failed: %s", exc)

    async def _resolve_user_item(self, text: str) -> Optional[int]:
        """Find the stored row for a just-noted user turn. The writer thread
        owns inserts, so the row lands a few ms after note_user_turn — a short
        poll beats wiring row ids back through the enqueue-only hot path."""
        for _ in range(30):
            if self.lifetime.closing:
                return None
            found = await asyncio.to_thread(self._find_user_item, text)
            if found is not None:
                return found
            await asyncio.sleep(0.1)
        return None

    def _find_user_item(self, text: str) -> Optional[int]:
        for item in self.store.recent(self.conversation_id, [KIND_UTTERANCE], limit=12):
            if item.role == "user" and (item.text or "").strip() == text:
                return item.id
        return None

    def _recent_context(self, exclude_id: int) -> List[str]:
        """Surrounding turns (chronological) to interpret an elliptical user
        turn. Assistant turns are CONTEXT ONLY — never a fact source."""
        out: List[str] = []
        for item in self.store.recent(self.conversation_id, [KIND_UTTERANCE], limit=8):
            if item.id == exclude_id or not item.text:
                continue
            role = "用户" if item.role == "user" else "助手"
            out.append(f"{role}: {item.text}")
        out.reverse()
        return out

    # ---- lifecycle ----

    @while_open(True)
    def has_history(self) -> bool:
        """Conservative absence check: pending/native context also counts."""
        return bool(self._native_context or self._injected or self._uncommitted or
                    self.store.count(self.conversation_id))

    @while_open(True)
    def has_dialogue_history(self) -> bool:
        return bool(self._native_context or self._injected or self._uncommitted or
                    self.store.recent(self.conversation_id, [KIND_UTTERANCE], limit=1))

    def cleared_copy(self) -> "MemorySession":
        """Drain the old lifetime before reusing this conversation's storage key."""
        self.close()
        fresh = MemorySession(self.conversation_id, self.settings, self.store, self.writer,
                              default_lang=self.language, facts=self._facts)
        fresh._t0 = self._t0  # video timestamps still use the original session clock
        return fresh

    def close(self) -> None:
        self.lifetime.seal()
        with self._close_lock:
            if self._cleaned:
                return
            self.writer.discard_session(self.conversation_id, self.lifetime)
            self.lifetime.wait()
            self.writer.forget(self.conversation_id)
            with self._lock:
                self._prefetch = None
                self._uncommitted.clear()
                self._injected.clear()
                self._native_context.clear()
                self._suppressed.clear()
                self._pending_injection_bodies.clear()
            count = self.store.delete_session(self.conversation_id)
            self._cleaned = True
            log.info("memory session cleaned: %s (%d items)", self.conversation_id, count)

    def status(self) -> Dict[str, Any]:
        return {"items": self.store.count(self.conversation_id),
                "resources": self.store.resource_status(self.conversation_id),
                "writer_budget_rejected": self.writer.stats.get('budget_rejected', 0),
                "injected_tokens": int(self._lifetime_tokens),
                "lang": self.language, **self.stats}
