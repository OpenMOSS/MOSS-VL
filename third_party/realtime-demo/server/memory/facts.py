"""Background fact extraction → retrieval index keys (design §3, §4).

Facts are NEVER shown to the model. They only enrich the retrieval index key
(`key = turn_text + facts`), so an elliptical later query ("那台相机多少钱") can
still hit the turn that introduced the camera even when the turn's own words
don't overlap the query. `memory_items.text` stays the raw verbatim turn —
retrieval returns raw text; the key lives in `memory_item_keys` for audit.

Extraction runs on the OFFLINE sglang plane (`rt.vlm_offline`), from USER turns
only (`memory_fact_scope == "user"`); assistant turns may be passed as context
to interpret an elliptical user turn but are never extracted from. Provider
!= "offline", a missing plane, or an unloaded plane all degrade to a silent
no-op (1-GPU boxes have no offline plane, design §7).

Concurrency: ONE semaphore (memory_bg_concurrency) shared with the future
rollover summarizer, so background jobs never stampede the sidecar. Nothing
here may run on the hot path — the orchestrator fires and forgets.
"""
from __future__ import annotations

import asyncio
from contextlib import nullcontext
import re
from typing import Any, List, Optional, Sequence

from ..config import Settings
from ..logging_conf import get_logger
from ..schemas import ChatMessage, ChatRequest, GenerationParams
from .store import SPACE_TEXT, SPACE_TEXT_LI, MemoryStore
from .writer import MemoryWriter

log = get_logger(__name__)

_MAX_FACTS = 8
# fact lines are short; the shared summary budget is the ceiling, not the target
_MAX_NEW_TOKENS = 160

_PROMPT = """从下面的用户发言中抽取原子事实(实体、属性、承诺、计划),每条一行短句;只输出事实行,不要编号解释,没有事实就输出空。
Extract atomic facts (entities, attributes, commitments, plans) from the USER turn below, one short line each; output ONLY the fact lines, no numbering or commentary, nothing if there are no facts.

对话上下文(仅用于理解指代,不要从中抽取)/ Context (reference resolution only, do NOT extract from it):
{context}

用户发言 / USER turn:
{text}"""

_LINE_BULLET_RE = re.compile(r"^\s*(?:[-*•·]+|\d+[.)、])\s*")


def parse_facts(text: str) -> List[str]:
    """Model output -> fact lines. Tolerant of bullets/numbering, caps count."""
    out: List[str] = []
    for line in (text or "").splitlines():
        line = _LINE_BULLET_RE.sub("", line).strip().strip("。.")
        if line and line not in out:
            out.append(line)
        if len(out) >= _MAX_FACTS:
            break
    return out


class FactExtractor:
    """Shared across sessions; every call carries its own conversation_id."""

    def __init__(self, settings: Settings, store: MemoryStore, writer: MemoryWriter,
                 plane: Any = None) -> None:
        self.settings = settings
        self.store = store
        self.writer = writer
        self.plane = plane
        # shared with (future) rollover summaries — design §7: background
        # concurrency 1-2 against the offline sidecar
        self.semaphore = asyncio.Semaphore(max(1, int(settings.memory_bg_concurrency)))

    def available(self) -> bool:
        if (self.settings.memory_summary_provider or "") != "offline":
            return False
        plane = self.plane
        if plane is None:
            return False
        try:
            return bool(plane.is_loaded())
        except Exception:  # noqa: BLE001
            return False

    async def extract_and_rekey(self, conversation_id: str, item_id: int, user_text: str,
                                context_turns: Optional[Sequence[str]] = None, *, lifetime: Any = None
                                ) -> Optional[str]:
        """Extract facts for one user turn and re-embed its vectors on the key.
        Returns the key, or None on any failure — the item then simply keeps
        retrieving on its raw text, which is the safe degradation."""
        text = (user_text or "").strip()
        if not text or not self.available() or (lifetime is not None and lifetime.closing):
            return None
        async with self.semaphore:
            if lifetime is not None and lifetime.closing:
                return None
            try:
                facts = await self._extract(text, context_turns or [])
            except Exception as exc:  # noqa: BLE001
                log.debug("memory fact extraction failed: %s", exc)
                return None
            if not facts:
                return None
            key = f"{text} {' '.join(facts)}"
            try:
                await asyncio.to_thread(self._rekey, conversation_id, item_id, key, lifetime=lifetime)
            except Exception as exc:  # noqa: BLE001
                log.debug("memory fact rekey failed: %s", exc)
                return None
            log.debug("memory: facts rekeyed item %d (%d facts)", item_id, len(facts))
            return key

    async def _extract(self, user_text: str, context_turns: Sequence[str]) -> List[str]:
        prompt = _PROMPT.format(context="\n".join(context_turns) or "(none)", text=user_text)
        req = ChatRequest(
            messages=[ChatMessage(role="user", content=prompt)],
            params=GenerationParams(max_new_tokens=_MAX_NEW_TOKENS, temperature=0.0))
        parts: List[str] = []
        async for delta in self.plane.generate_stream(req):
            parts.append(delta)
        return parse_facts("".join(parts))

    def _rekey(self, conversation_id: str, item_id: int, key: str, *, lifetime: Any = None) -> None:
        with lifetime.operation() if lifetime is not None else nullcontext(True) as admitted:
            if admitted:
                self._rekey_open(conversation_id, item_id, key)

    def _rekey_open(self, conversation_id: str, item_id: int, key: str) -> None:
        """Replace the item's vectors with encodings of the KEY and persist the
        key for audit. `memory_items.text` is deliberately untouched — the raw
        verbatim turn is what retrieval returns."""
        if not self.store.accepting_writes:
            return
        embedder = self.writer.text
        self.store.update_vector(conversation_id, item_id, SPACE_TEXT, embedder.encode([key])[0])
        encode_tokens = getattr(embedder, "encode_tokens", None)
        if self.settings.memory_late_interaction and getattr(embedder, "supports_late", False) \
                and callable(encode_tokens):
            try:
                self.store.update_vector(conversation_id, item_id, SPACE_TEXT_LI,
                                         encode_tokens([key])[0])
            except Exception as exc:  # noqa: BLE001 — pooled rekey already landed
                log.debug("memory fact rekey (late) failed: %s", exc)
        self.store.put_key(item_id, key, conversation_id=conversation_id)


def build_fact_extractor(settings: Settings, store: Optional[MemoryStore],
                         writer: Optional[MemoryWriter], plane: Any = None
                         ) -> Optional[FactExtractor]:
    """The extractor for the Runtime, or None when memory is off — construction
    never fails boot; an absent/unloaded plane is a runtime no-op instead."""
    if store is None or writer is None:
        return None
    try:
        return FactExtractor(settings, store, writer, plane=plane)
    except Exception as exc:  # noqa: BLE001
        log.warning("memory facts disabled: %s", exc)
        return None
