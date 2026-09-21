"""Matching: fuse the two spaces, rank, diversify, and decide whether to inject.

The gate is deliberately skip-biased. Every injected token is a permanent
resident of the session KV cache — it cannot be removed, and a *related but
wrong* memory is the worst noise class there is (measurably worse than no
context at all, because it looks like evidence). So the default answer is "do
not inject": a turn must either carry an explicit past-reference marker and
clear the score threshold, or clear a distinctly higher threshold on its own.

Ranking is the Generative-Agents blend — relevance x recency-decay x importance
— with an MMR pass so two near-identical memories never both burn budget, and a
similarity check against the query itself so we never "recall" what the user
just said.

Fusion uses per-space Z-NORMALIZATION against a reservoir of raw scores seen in
past searches (design §4: raw CLIP and BGE cosines are not comparable, so each
space is normalized against its own background distribution); below 30 samples
a space falls back to per-hit-set min-max rather than normalize against a noise
estimate. The admission gate never reads either normalized value — it reads the
ABSOLUTE raw cosine/maxsim only (see Candidate.raw).

`memory_reranker` is a documented no-op (design §1: no reranker); the config
key exists only so the config surface matches the design doc.
"""
from __future__ import annotations

import random
import re
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set

import numpy as np

from ..config import Settings
from ..logging_conf import get_logger
from . import lang as lang_mod
from .store import KIND_CAPTION, KIND_FRAME, KIND_PINNED, SPACE_IMAGE, SPACE_TEXT, MemoryItem, MemoryStore

log = get_logger(__name__)

# explicit "we talked about this before" markers, per language
_PAST_MARKERS_ZH = ("刚才", "刚刚", "之前", "上次", "还记得", "记得", "我说过", "我讲过",
                    "给你看", "给你看过", "先前", "早些", "前面", "方才", "那个东西", "上回",
                    "最开始", "一开始", "最早", "开头")
_PAST_MARKERS_EN = ("earlier", "before", "last time", "remember", "recall", "previously",
                    "you saw", "i showed", "i told you", "that one", "we discussed")
# ASR transcripts carry no punctuation, so Chinese questions are detected by
# interrogative words anywhere in the text; English stays anchored at the head
_QUESTION_RE = re.compile(r"[?？吗呢]|什么|啥|哪|谁|多少|怎么|怎样|为何"
                          r"|^(what|when|where|who|which|how|did|do|does|is|are|was|were)\b",
                          re.IGNORECASE)
# a background sample smaller than this is a noise estimate, not a distribution
_ZNORM_MIN_SAMPLES = 30


@dataclass
class Candidate:
    item: MemoryItem
    relevance: float          # fused, per-space z-normed (min-max fallback)
    score: float = 0.0        # final ranked score (ordering only)
    space: str = SPACE_TEXT
    late: bool = False        # raw is a colbert maxsim mean, not a pooled cosine
    raw: float = 0.0          # ABSOLUTE cosine/maxsim in its own space — the
                              # gate reads this, never `relevance`: normalization
                              # maps the best of a bad set high either way, so
                              # gating on it would admit the top hit of every turn.


def has_past_reference(text: str) -> bool:
    low = (text or "").lower()
    if any(m in text for m in _PAST_MARKERS_ZH):
        return True
    return any(m in low for m in _PAST_MARKERS_EN)


def is_history_question(text: str) -> bool:
    return has_past_reference(text) and bool(_QUESTION_RE.search(text or ''))


def refers_to_dialogue(text: str) -> bool:
    """Distinguish an explicit prior conversation from a prior camera view."""
    return bool(re.search(r'(?:我|你|我们).{0,12}(?:说|告诉|提到|问|记住|记得)', text or '')) or bool(
        re.search(r'\b(?:i|you|we)\b.{0,40}\b(?:said|told|asked|mentioned|remember)\b', text or '', re.I))


def _minmax(values: Sequence[float]) -> List[float]:
    """Per-space min-max — the FALLBACK fusion normalizer while a space has
    fewer than _ZNORM_MIN_SAMPLES background observations."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-6:
        return [1.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


class Retriever:
    def __init__(self, store: MemoryStore, text_embedder, image_embedder,
                 settings: Settings) -> None:
        self.store = store
        self.text = text_embedder
        self.image = image_embedder
        self.settings = settings
        # per-space reservoirs of RAW scores seen in searches — the background
        # distribution the z-norm fusion normalizes against
        self._bg: Dict[str, List[float]] = {}
        self._bg_seen: Dict[str, int] = {}
        self._rng = random.Random(0xB6E)

    def _observe_scores(self, space: str, scores: Sequence[float]) -> None:
        """Reservoir-sample raw hit scores (capped at memory_znorm_samples)."""
        cap = max(_ZNORM_MIN_SAMPLES, int(self.settings.memory_znorm_samples))
        buf = self._bg.setdefault(space, [])
        seen = self._bg_seen.get(space, 0)
        for score in scores:
            seen += 1
            if len(buf) < cap:
                buf.append(float(score))
            else:
                j = self._rng.randrange(seen)
                if j < cap:
                    buf[j] = float(score)
        self._bg_seen[space] = seen

    def _normalize(self, space: str, vals: Sequence[float]) -> List[float]:
        """Z against the background sample; min-max while it is too small."""
        buf = self._bg.get(space) or []
        if len(buf) >= _ZNORM_MIN_SAMPLES:
            arr = np.asarray(buf, dtype=np.float64)
            std = float(arr.std())
            if std >= 1e-6:
                mean = float(arr.mean())
                return [(float(v) - mean) / std for v in vals]
        return _minmax(vals)

    # ---- search ----

    def search(self, conversation_id: str, query: str, *, exclude: Optional[Set[int]] = None,
               limit: int = 8) -> List[Candidate]:
        """Top-k across the text space (+ image space when it is cross-modal)."""
        query = (query or "").strip()
        if not query:
            return []
        started = time.perf_counter()
        qvec = self.text.encode([query])[0]
        dense_encoded = time.perf_counter()
        hits = self.store.search(conversation_id, SPACE_TEXT, qvec,
                                 limit=limit * 2, exclude=exclude)
        dense_searched = time.perf_counter()
        pairs: List[tuple[int, float, str]] = [(i, s, SPACE_TEXT) for i, s in hits]

        # late interaction (design §4): where a text_li matrix exists, its
        # maxsim mean REPLACES the pooled cosine as the item's text-space raw.
        # Pooled search still runs — it discovers items written before the lane
        # was enabled (or without token matrices at all).
        late_scores: Dict[int, float] = {}
        late_encoded = dense_searched
        encode_tokens = getattr(self.text, "encode_tokens", None)
        if self.settings.memory_late_interaction and callable(encode_tokens):
            try:
                qtok = encode_tokens([query])[0]
                late_encoded = time.perf_counter()
                late_scores = dict(self.store.search_late(conversation_id, qtok,
                                                          limit=limit * 2, exclude=exclude))
            except Exception as exc:  # noqa: BLE001 — pooled retrieval still works
                log.debug("memory late-interaction search failed: %s", exc)
        if late_scores:
            pooled_ids = {i for i, _, _ in pairs}
            pairs = [(i, late_scores.get(i, s), sp) for i, s, sp in pairs]
            pairs.extend((i, s, SPACE_TEXT) for i, s in late_scores.items() if i not in pooled_ids)
        late_searched = time.perf_counter()

        # frames are reachable by a text query only through a cross-modal image
        # embedder; with the fallback descriptor they are recalled via captions
        if getattr(self.image, "cross_modal", False):
            ivec = self.image.encode_texts([query])[0]  # type: ignore[attr-defined]
            for item_id, score in self.store.search(conversation_id, SPACE_IMAGE, ivec,
                                                    limit=limit, exclude=exclude):
                pairs.append((item_id, score, SPACE_IMAGE))

        image_searched = time.perf_counter()
        log.info('memory search session=%s dense_encode_ms=%.1f dense_search_ms=%.1f '
                 'late_encode_ms=%.1f late_search_ms=%.1f image_ms=%.1f', conversation_id,
                 (dense_encoded-started)*1000, (dense_searched-dense_encoded)*1000,
                 (late_encoded-dense_searched)*1000, (late_searched-late_encoded)*1000,
                 (image_searched-late_searched)*1000)
        if not pairs:
            return []
        by_space = {}
        for space in {p[2] for p in pairs}:
            vals = [p[1] for p in pairs if p[2] == space]
            self._observe_scores(space, vals)
            norm = self._normalize(space, vals)
            by_space[space] = dict(zip([p[0] for p in pairs if p[2] == space], norm))

        items = self.store.get_items([p[0] for p in pairs])
        w_text = float(self.settings.memory_fusion_text_weight)
        best: dict[int, Candidate] = {}
        for item_id, raw, space in pairs:
            item = items.get(item_id)
            if item is None or item.invalid_at is not None:
                continue
            weight = w_text if space == SPACE_TEXT else (1.0 - w_text)
            rel = by_space[space].get(item_id, 0.0) * weight
            cur = best.get(item_id)
            if cur is None or rel > cur.relevance:
                best[item_id] = Candidate(item=item, relevance=rel, space=space,
                                          late=(space == SPACE_TEXT and item_id in late_scores),
                                          raw=max(float(raw), cur.raw if cur else 0.0))
        return list(best.values())

    def search_visual(self, conversation_id: str, jpeg: bytes, *, limit: int = 5) -> List[Candidate]:
        """Find past keyframes that look like this one ('have I seen this?')."""
        vec = self.image.encode_images([jpeg])[0]
        hits = self.store.search(conversation_id, SPACE_IMAGE, vec, limit=limit)
        items = self.store.get_items([i for i, _ in hits])
        out = []
        for item_id, score in hits:
            item = items.get(item_id)
            if item is not None:
                out.append(Candidate(item=item, relevance=float(score), score=float(score),
                                     space=SPACE_IMAGE))
        return out

    # ---- rank ----

    def rank(self, candidates: List[Candidate], *, now_session_ts: Optional[float] = None
             ) -> List[Candidate]:
        half_life_s = max(60.0, float(self.settings.memory_recency_halflife_h) * 3600.0)
        now = time.time()
        for cand in candidates:
            # Decay from CREATION, deliberately not from last-retrieval. The
            # Generative Agents formula this blend is modelled on decays from last
            # access, so retrieving a memory makes it rank higher next time. That is
            # right for a stateless store and wrong here: our KV is append-only, so a
            # retrieval-strengthens-recency loop would spend a permanent, unreclaimable
            # token budget re-showing whatever we just showed. Looks like a bug, isn't.
            age = max(0.0, now - (cand.item.created_at or now))
            recency = 0.5 ** (age / half_life_s)
            bonus = 0.15 if cand.item.kind == KIND_PINNED else 0.0
            cand.score = (cand.relevance
                          + 0.25 * recency
                          + 0.30 * float(cand.item.importance or 0.5)
                          + bonus)
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates

    def diversify(self, candidates: List[Candidate], query: str, *, limit: int,
                  sim_cutoff: float = 0.86) -> List[Candidate]:
        """MMR-ish: drop items too similar to the query or to an already-picked one."""
        if not candidates:
            return []
        texts = [c.item.text or "" for c in candidates]
        vecs = self.text.encode([query] + texts)
        qv, cv = vecs[0], vecs[1:]
        picked: List[Candidate] = []
        picked_idx: List[int] = []
        for i, cand in enumerate(candidates):
            if float(cv[i] @ qv) >= sim_cutoff:
                continue  # the user just said this — recalling it adds nothing
            if any(float(cv[i] @ cv[j]) >= sim_cutoff for j in picked_idx):
                continue
            picked.append(cand)
            picked_idx.append(i)
            if len(picked) >= limit:
                break
        return picked

    # ---- gate ----

    def gate(self, query: str, candidates: List[Candidate]) -> List[Candidate]:
        """Skip-biased admission. Returns [] far more often than not."""
        if not candidates:
            return []
        explicit = has_past_reference(query)
        # a turn with no explicit past-reference marker must clear a distinctly
        # higher bar; one that is not even a question, higher still
        margin = 0.0 if explicit else 0.15
        if not explicit and not _QUESTION_RE.search(query or ""):
            margin += 0.10
        keep = []
        for cand in candidates:
            # per-SPACE floor: a CLIP cosine (~0.1-0.35) and a text-encoder
            # cosine (~0.5-0.8) are not comparable, so one global threshold
            # would either mute the visual lane or open the text lane wide
            base = float(self.settings.memory_gate_min_score)
            if base <= 0.0:
                if cand.space == SPACE_IMAGE:
                    base = float(getattr(self.image, "gate_floor", 0.30))
                elif cand.late:
                    # maxsim means sit higher than pooled cosines — own floor,
                    # calibration-pending per design §10
                    base = float(getattr(self.text, "li_gate_floor", 0.5))
                else:
                    base = float(getattr(self.text, "gate_floor", 0.30))
            # the margin is in cosine units, so it shrinks with the space's range
            mscale = 1.0 if cand.space == SPACE_TEXT else 0.5
            if cand.raw >= base + margin * mscale:
                keep.append(cand)
        return keep[: max(0, int(self.settings.memory_inject_max_items))]

    # ---- one-shot convenience used by the session ----

    def recall(self, conversation_id: str, query: str, *, exclude: Optional[Set[int]] = None,
               ) -> List[Candidate]:
        limit = max(1, int(self.settings.memory_retrieval_topk))
        found = self.search(conversation_id, query, exclude=exclude, limit=limit)
        if not found:
            return []
        ranked = self.rank(found)
        diverse = self.diversify(ranked, query, limit=limit)
        return self.gate(query, diverse)
