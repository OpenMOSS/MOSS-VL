"""Per-item language decision (zh | en) — deterministic, never model-mediated.

Why a hand-rolled rule instead of langdetect/fastText: the decision here is
binary (zh|en) over 5-50 char fragments, which is exactly where statistical LID
degrades (langdetect is non-deterministic on short strings; fastText skews
English on short Latin fragments). Once the label set is script-separable, a
CJK-character ratio *is* the classifier — CLD2/CLD3 partition by script before
any statistics for the same reason.

Threshold: code-switched zh-en follows the matrix-language pattern (Chinese
frame, embedded English jargon: "帮我看看这个 error message"), and a hanzi carries
~2-3x the information of a latin letter, so the rule is zh-biased — 0.25 CJK
ratio is enough to call the fragment Chinese.

The rolling `LanguageState` exists because captions have no source language of
their own: they inherit the conversation's current language. It is causal (only
past turns) and hysteretic (two consecutive turns to flip), so a single English
noun cannot swing caption language for the rest of the session.
"""
from __future__ import annotations

from typing import Optional

ZH = "zh"
EN = "en"

# CJK ideographs + ext-A + compatibility + CJK punctuation/fullwidth forms
_CJK_RANGES = (
    (0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0xF900, 0xFAFF),
    (0x3000, 0x303F), (0xFF00, 0xFFEF),
)
# zh unless the fragment is overwhelmingly latin (see module docstring)
CJK_RATIO_THRESHOLD = 0.25
# fragments shorter than this carry no reliable signal ("ok", "嗯", emoji)
MIN_SIGNAL_CHARS = 5


def _is_cjk(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _CJK_RANGES)


def cjk_ratio(text: str) -> float:
    """CJK share of the letter mass; digits/punctuation/space are ignored."""
    cjk = latin = 0
    for ch in text or "":
        if _is_cjk(ch):
            cjk += 1
        elif ch.isalpha():
            latin += 1
    total = cjk + latin
    return (cjk / total) if total else 0.0


def detect_lang(text: str, default: str = ZH) -> str:
    """zh|en for one fragment. `default` covers text with no letters at all."""
    cjk = latin = 0
    for ch in text or "":
        if _is_cjk(ch):
            cjk += 1
        elif ch.isalpha():
            latin += 1
    if cjk + latin == 0:
        return default
    return ZH if (cjk / (cjk + latin)) >= CJK_RATIO_THRESHOLD else EN


def has_signal(text: str) -> bool:
    """Enough letters to be worth a language vote."""
    return sum(1 for ch in (text or "") if _is_cjk(ch) or ch.isalpha()) >= MIN_SIGNAL_CHARS


def dominant_lang(texts, default: str = ZH) -> str:
    """Language of a multi-turn span, by character mass (not turn count)."""
    cjk = latin = 0
    for text in texts:
        for ch in text or "":
            if _is_cjk(ch):
                cjk += 1
            elif ch.isalpha():
                latin += 1
    if cjk + latin == 0:
        return default
    return ZH if (cjk / (cjk + latin)) >= CJK_RATIO_THRESHOLD else EN


class LanguageState:
    """Rolling conversation language: what captions are written in.

    Fed only by USER turns (assistant text mirrors whatever it was asked in, so
    it would double-count). Flips only after `switch_after` consecutive turns in
    the other language, so one English sentence in a Chinese session does not
    re-language the captions.
    """

    def __init__(self, default: str = ZH, switch_after: int = 2) -> None:
        self.current = default
        self._switch_after = max(1, switch_after)
        self._streak = 0
        self._streak_lang: Optional[str] = None

    def observe(self, text: str) -> str:
        """Feed one user turn; returns the (possibly unchanged) current language."""
        if not has_signal(text):
            return self.current
        lang = detect_lang(text, default=self.current)
        if lang == self.current:
            self._streak = 0
            self._streak_lang = None
            return self.current
        if lang == self._streak_lang:
            self._streak += 1
        else:
            self._streak_lang = lang
            self._streak = 1
        if self._streak >= self._switch_after:
            self.current = lang
            self._streak = 0
            self._streak_lang = None
        return self.current
