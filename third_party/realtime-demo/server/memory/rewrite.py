"""Rule-based pre-retrieval query rewriting (design §4).

Deictic and relative-time queries ("刚才那个是什么", "what was that one I showed
you") fail under raw embedding similarity: the informative token is the deictic
noun, which matches nothing, while the actual referent sits a few turns back.
So these two recoverable cases are resolved by RULES before retrieval, never by
embedding similarity alone:

- relative-time phrases parse to a session-time window that filters (or, when
  the window is empty, leaves untouched) candidates by `session_ts`;
- a short deictic past-reference query is AUGMENTED with the most recent user
  turns, giving the embedder real content to bite on.

Everything here is a pure function and conservative by contract: anything
unrecognized passes through untouched. A wrong rewrite costs more than no
rewrite — the same skip-bias as the gate.
"""
from __future__ import annotations

import re
from typing import List, Optional, Sequence, Tuple

from .retrieval import has_past_reference

# (lo, hi) in SESSION seconds, as stored on memory_items.session_ts
TimeWindow = Tuple[float, float]

# deictic nouns that signal "the thing I mentioned/showed, not a new topic"
_DEICTIC_ZH = ("那个东西", "那台", "那个", "那件", "那部", "这台", "这个")
_DEICTIC_EN = ("that one", "this one", "that", "this")
# augmentation fires only on SHORT queries: a long query carries its own
# content and appending old turns would just dilute it
_DEICTIC_MAX_CHARS = 40

# marker -> window shape, checked in order (most specific first)
_JUST_NOW_ZH = ("刚才", "刚刚", "方才")
_JUST_NOW_EN = ("just now",)
_EARLIER_ZH = ("早些时候", "之前", "先前")
_EARLIER_EN = ("earlier", "previously", "before")
_LAST_TIME_ZH = ("上次", "上回")
_LAST_TIME_EN = ("last time",)

# half-width of a pointed window ("3分钟前" means "around then", not an instant)
_HALF_WIDTH_S = {"秒": 20.0, "分": 60.0, "时": 600.0}
# "the very start of the conversation" — chronological, not semantic: the
# referent sits at session_ts ≈ 0 and embedding similarity can never find it
# ("你好" vs "我最开始说了什么" share no content), so restrict by time
_SESSION_START_ZH = ("最开始", "一开始", "第一句", "开头")
_SESSION_START_EN = ("first thing", "very first", "at the beginning", "at the start",
                     "beginning of")
_SESSION_START_WINDOW_S = 60.0


def temporal_order(text: str) -> Optional[str]:
    """Event order is distinct from a fixed window at session start."""
    if not re.search(r'什么|哪|几|回忆|告诉|[?？]|\b(?:what|which|when|recall|show)\b', text or '', re.I):
        return None
    if re.search(r'最早|首次|第一(?:个|次)|第1(?:个|次)|最先|最初|\b(?:first|earliest)\b', text or '', re.I):
        return 'earliest'
    if re.search(r'最后(?:一次)?|最近一次|\b(?:last|latest|most recent)\b', text or '', re.I):
        return 'latest'
    return None
_ZH_AGO_RE = re.compile(r"(\d{1,4})\s*(秒|分钟?|个?小时|个?钟头)前")
_EN_AGO_RE = re.compile(r"\b(\d{1,4})\s*(seconds?|secs?|minutes?|mins?|hours?|hrs?)\s+ago\b",
                        re.IGNORECASE)
# explicit stream stamp, matching the recall-line format "[t=1:33]"
_STAMP_RE = re.compile(r"\bt=(\d{1,4}):(\d{2})\b")


def _pointed(center: float, half_width: float) -> Optional[TimeWindow]:
    if center < 0:
        return None  # "3分钟前" 20 s into a session points at nothing
    return (max(0.0, center - half_width), center + half_width)


def parse_time_window(text: str, now_session_ts: float) -> Optional[TimeWindow]:
    """Relative-time phrase -> session-time window, or None (pass through).

    `now_session_ts` is the caller's session clock; the window is expressed in
    the same units as memory_items.session_ts so candidates can be filtered
    directly.
    """
    text = (text or "").strip()
    if not text:
        return None
    now = max(0.0, float(now_session_ts))
    low = text.lower()

    stamp = _STAMP_RE.search(low)
    if stamp:
        ts = int(stamp.group(1)) * 60 + int(stamp.group(2))
        return (max(0.0, ts - 30.0), ts + 30.0)

    # session-start references first: "最开始" contains none of the markers
    # below, but ordering it up front keeps the intent unmistakable
    if any(m in text for m in _SESSION_START_ZH) or any(m in low for m in _SESSION_START_EN):
        return (0.0, min(_SESSION_START_WINDOW_S, now))

    ago = _ZH_AGO_RE.search(text)
    if ago:
        n = float(ago.group(1))
        unit = ago.group(2)
        scale = 1.0 if unit.startswith("秒") else (3600.0 if ("时" in unit or "钟头" in unit) else 60.0)
        half = _HALF_WIDTH_S["秒" if scale == 1.0 else ("时" if scale == 3600.0 else "分")]
        return _pointed(now - n * scale, half)

    ago = _EN_AGO_RE.search(low)
    if ago:
        n = float(ago.group(1))
        unit = ago.group(2)
        scale = 1.0 if unit.startswith(("s",)) else (3600.0 if unit.startswith("h") else 60.0)
        half = _HALF_WIDTH_S["秒" if scale == 1.0 else ("时" if scale == 3600.0 else "分")]
        return _pointed(now - n * scale, half)

    if any(m in text for m in _JUST_NOW_ZH) or any(m in low for m in _JUST_NOW_EN):
        return (max(0.0, now - 300.0), now)
    if any(m in text for m in _EARLIER_ZH) or any(m in low for m in _EARLIER_EN):
        return (0.0, max(0.0, now - 30.0))
    if any(m in text for m in _LAST_TIME_ZH) or any(m in low for m in _LAST_TIME_EN):
        return (0.0, max(0.0, now - 60.0))
    return None


def is_deictic_query(text: str) -> bool:
    """A SHORT past-reference query built around a deictic noun — the case where
    augmentation is safe and raw similarity is hopeless."""
    text = (text or "").strip()
    if not text or len(text) > _DEICTIC_MAX_CHARS:
        return False
    low = text.lower()
    if not (any(n in text for n in _DEICTIC_ZH) or any(n in low for n in _DEICTIC_EN)):
        return False
    return has_past_reference(text)


def augment_query(text: str, recent_user_texts: Sequence[str]) -> str:
    """Append recent user turns so a deictic query retrieves its referent.
    The query's own text (if it was already indexed) is never re-appended."""
    query = (text or "").strip()
    extra: List[str] = []
    for turn in recent_user_texts:
        turn = (turn or "").strip()
        if turn and turn != query:
            extra.append(turn)
    return " ".join([query, *extra]) if extra else query
