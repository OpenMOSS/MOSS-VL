"""Rollover (compaction) — design §6.

MOSS-VL's text KV is append-only and grows forever; ~80% of that growth is
timestamp wrappers for frames whose vision KV the frame window already
evicted. When the worker's exact TEXT-token count (the token-counter patch in
realtime/mossvl_patches.py, surfaced via the 1 Hz status) crosses a threshold,
this module rebuilds the prefix and the orchestrator re-seats the session:

- idle trigger `memory_rollover_idle_tokens`, only at a `<|silence|>` idle moment
- hard trigger `memory_rollover_hard_tokens`, fires regardless (but never
  mid-streaming-assistant-output — the orchestrator defers a live response)
- anti-thrash floor: skip when less than `memory_rollover_min_progress` of the
  text KV would actually be reclaimed

The new prefill is ONE role:system message at position 0 — base prompt /
recall-format declaration / pinned / verbatim-hold / summary / handle line —
followed by the last `memory_rollover_tail_turns` turns verbatim as real
user/assistant tail messages. The summary is RE-DERIVED from the full raw
journal each rollover (bounded QA chunks are merged within that job) — on the pi_agent sidecar
(provider "pi": /compact → {summary, pins}, pins join the pinned layer) or on
the offline sglang plane (provider "offline"); any other provider or an
absent/unloaded/failing backend degrades to verbatim-tail-only with NO error
(1-GPU boxes have no offline plane, §7).
Correction detection and assembly are lexical/string-concat — no model (§7).
"""
from __future__ import annotations

import asyncio
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..config import Settings
from ..logging_conf import get_logger
from ..schemas import ChatMessage, ChatRequest, GenerationParams
from . import inject as inject_mod
from .pi_client import PiAgentClient
from .lifecycle import SessionLifetime, while_open
from .store import KIND_PINNED, KIND_UTTERANCE, MemoryItem, MemoryStore

log = get_logger(__name__)

# Layer token budgets in estimate_tokens units (over-estimating is the safe
# direction). Target prefix ≈1.5k tokens — design §6's headroom math assumes it.
_BUDGET_PINNED = 240         # user-pinned items; the last layer ever evicted
_BUDGET_VERBATIM_HOLD = 320  # corrections / identifiers / numbers / commitments
_BUDGET_TAIL = 700           # the verbatim recent-turns tail
# base prompt + recall declaration + handle line are fixed scaffolding and are
# estimated from their real text, not budgeted.

_JOURNAL_LIMIT = -1          # all raw utterances; never silently omit older history
_TAIL_MSG_OVERHEAD = 8       # chat-template scaffolding per tail message (est.)

# verbatim-hold extraction (lexical, design §6 — no model):
_CORRECTION_RE = re.compile(r"(不是那个|不是|我说的是|我是说|I meant|I said)", re.IGNORECASE)
_NUMBER_UNIT_RE = re.compile(
    r"\d+(?:\.\d+)?\s?(?:元|块|岁|年|月|日|号|点|分|秒|米|厘米|毫米|公里|千克|克|斤|升|毫升"
    r"|km|cm|mm|kg|mg|GB|MB|TB|mL|ml|°C|°F|%|dollars?|years?|hours?|minutes?|seconds?|meters?|miles?)")
# proper nouns / identifiers: hyphen-joined latin runs (BGE-M3), ALLCAPS+digits
# (FM2), or letter+digit mixes (GPT4) — the things a summarizer paraphrases away
_IDENTIFIER_RE = re.compile(r"[A-Za-z][\w.]*(?:[-_][\w.]+)+|[A-Z]{2,}\d+|\b[A-Za-z]+\d+\b")
_COMMITMENT_RE = re.compile(r"(我会|我帮你|我这就|我来帮|I will|I'll|let me)", re.IGNORECASE)

_SUMMARY_PROMPT = """把下面的实时对话压缩成一段简短摘要,按时间顺序保留事实、决定、待办和专有名称;只输出摘要正文,不要编号或解释。
Summarize the realtime conversation below, chronological; keep facts, decisions, todos and proper names; output ONLY the summary text, no numbering or commentary.

对话 / Conversation:
{journal}"""

# mirror of mossvl_patches.DEFAULT_BOARD_REALTIME_SYSTEM_PROMPT — used only
# when the lazy import fails (a slim gateway venv without torch); keep in sync
_DEFAULT_SYSTEM_PROMPT_MIRROR = (
    "You are a helpful AI assistant specializing in real-time video analysis. "
    "The video streams to you frame by frame. At every frame, you decide independently "
    "whether to respond or stay silent — output `<|silence|>` when nothing relevant has happened, "
    "and respond when the visual content warrants it."
)


def _default_system_prompt() -> str:
    """Lazy import: realtime.mossvl_patches pulls in torch, which the gateway
    process does not otherwise need."""
    try:
        from ..realtime.mossvl_patches import DEFAULT_BOARD_REALTIME_SYSTEM_PROMPT
        return DEFAULT_BOARD_REALTIME_SYSTEM_PROMPT
    except Exception:  # noqa: BLE001
        return _DEFAULT_SYSTEM_PROMPT_MIRROR


@dataclass
class _Prefetch:
    """One background _collect + pi /compact run kicked ahead of the rollover
    thresholds (board parity); consumed by build_prefix."""
    done: threading.Event = field(default_factory=threading.Event)
    status: str = "running"  # running | ready | error
    result: Optional[Tuple[Tuple[List[MemoryItem], List[MemoryItem], List[MemoryItem]],
                           str, List[str]]] = None  # (collected, summary, pins)


class RolloverManager:
    """Per-session compaction planner. Construction is cheap; the orchestrator
    drives `should_rollover` (sync, hot-ish) and `build_prefix` (async, off the
    turn path). Every failure degrades to "no rollover", never a broken turn.
    """

    def __init__(self, settings: Settings, store: MemoryStore, conversation_id: str,
                 *, plane: Any = None, base_system_prompt: str = "",
                 lang_getter: Optional[Callable[[], str]] = None,
                 semaphore: Optional[asyncio.Semaphore] = None,
                 pi: Any = None, lifetime: Optional[SessionLifetime] = None,
                 journal_extra: Optional[Callable[[], Sequence[Tuple[str, str]]]] = None) -> None:
        self.settings = settings
        self.lifetime = lifetime or SessionLifetime()
        self.store = store
        self.conversation_id = conversation_id
        # offline sglang plane (rt.vlm_offline) or None; the summarizer shares
        # the background-job semaphore with fact extraction (design §7:
        # concurrency 1-2 against the sidecar), falling back to its own
        self.plane = plane
        # pi_agent sidecar client for memory_summary_provider == "pi"
        # (/compact → {summary, pins}); stateless, cheap to construct
        self.pi = pi if pi is not None else PiAgentClient(settings)
        # interrupted (uncommitted) turns from the MemorySession — compact
        # journal context only, never the tail (board QA-pair parity)
        self._journal_extra = journal_extra
        self.base_system_prompt = base_system_prompt or ""
        self._lang_getter = lang_getter or (lambda: "zh")
        self._semaphore = semaphore or asyncio.Semaphore(
            max(1, int(settings.memory_bg_concurrency)))
        # compact prefetch: one in-flight/ready background run per session
        self._prefetch_lock = threading.Lock()
        self._prefetch: Optional[_Prefetch] = None

    # ------------------------------------------------------------------ trigger

    def enabled(self) -> bool:
        return int(self.settings.memory_rollover_hard_tokens) > 0

    def should_rollover(self, text_tokens: Any, *, idle: bool = False) -> bool:
        """Trigger decision on the worker's exact text-token count.

        The hard trigger fires regardless of idleness; the idle trigger only at
        a `<|silence|>` idle moment. Both are subject to the anti-thrash floor:
        if the rebuilt prefix would carry almost as much as the current one,
        the seam costs more than it reclaims.
        """
        if not self.enabled():
            return False
        try:
            tokens = float(text_tokens)
        except (TypeError, ValueError):
            return False
        if tokens < float(self.settings.memory_rollover_hard_tokens):
            if not idle or tokens < float(self.settings.memory_rollover_idle_tokens):
                return False
        # anti-thrash: estimate what the new prefix will carry from CURRENT
        # journal sizes (fixed scaffolding + the budgeted layers at their real
        # sizes + the summary budget when a summary plane is configured). Only
        # an order-of-magnitude check — the real count is known post-build.
        try:
            _, _, est_prefix = self._assemble(summary=None)
        except Exception as exc:  # noqa: BLE001 — a broken estimate skips, never crashes
            log.warning("rollover estimate failed: %s", exc)
            return False
        if self._summary_configured():
            est_prefix += max(0, int(self.settings.memory_summary_max_tokens))
        if (tokens - est_prefix) < float(self.settings.memory_rollover_min_progress) * tokens:
            log.info("rollover skipped (anti-thrash): tokens=%d est_prefix=%d",
                     int(tokens), est_prefix)
            return False
        return True

    # ------------------------------------------------------------------ prefix build

    @while_open(([], [], 0))
    async def build_prefix(self, *, require_summary: bool = False) -> Tuple[List[dict], List[int], int]:
        """(prefill_messages, kept_item_ids, est_prefix_tokens).

        `kept_item_ids` are EXACTLY the journal ids whose content went into the
        new prefix — the orchestrator hands them to `MemorySession.note_rollover`
        so the injected set is recomputed from reality (design §5), not cleared.

        A ready compact prefetch (maybe_prefetch_compact) is consumed first —
        the seconds-long pi /compact call is normally done by the time the
        rollover fires; a still-running one gets a bounded join (30s), and a
        failed/empty one falls through to the synchronous path unchanged.
        """
        cached = await asyncio.to_thread(self._take_prefetch)
        journal, pinned, tail = await asyncio.to_thread(self._collect)
        if cached is not None:
            (old_journal, _, _), summary, pins = cached
            try:
                has_extra = bool(self._journal_extra and self._journal_extra())
            except Exception:
                has_extra = True
            if old_journal == journal and not has_extra and (summary or not require_summary):
                return self._assemble(summary, pins=pins, collected=(journal, pinned, tail))
            log.info("rollover compact prefetch stale; rebuilding from current journal")
        summary, pins = await self._summarize_full(journal)
        if require_summary and not summary:
            raise RuntimeError("No summary produced; existing context was not replaced")
        return self._assemble(summary, pins=pins, collected=(journal, pinned, tail))

    def cleared_copy(self, memory) -> "RolloverManager":
        return RolloverManager(
            self.settings, self.store, self.conversation_id, plane=self.plane,
            base_system_prompt=self.base_system_prompt, semaphore=self._semaphore,
            pi=self.pi, lifetime=memory.lifetime, lang_getter=lambda: memory.language,
            journal_extra=memory.uncommitted_turns)

    # ------------------------------------------------------------------ internals

    def _lang(self) -> str:
        try:
            return (self._lang_getter() or "zh").lower()
        except Exception:  # noqa: BLE001
            return "zh"

    def _base_prompt(self) -> str:
        return self.base_system_prompt.strip() or _default_system_prompt()

    def _collect(self) -> Tuple[List[MemoryItem], List[MemoryItem], List[MemoryItem]]:
        """(journal, pinned, tail), all chronological. Blocking sqlite — the
        async callers go through to_thread; `should_rollover` calls it directly
        from the orchestrator's status/idle hooks (a bounded indexed read)."""
        journal = list(reversed(self.store.recent(
            self.conversation_id, [KIND_UTTERANCE], limit=_JOURNAL_LIMIT)))
        pinned = list(reversed(self.store.recent(
            self.conversation_id, [KIND_PINNED], limit=64)))
        return journal, pinned, journal[self._tail_start(journal):]

    def _tail_start(self, journal: Sequence[MemoryItem]) -> int:
        """Boundary rule (design §6): complete QA pairs only — the kept tail
        must not begin with an assistant reply whose user turn is being
        compacted away, so extend the cut backwards past leading replies."""
        n = max(1, int(self.settings.memory_rollover_tail_turns))
        start = max(0, len(journal) - n)
        while start > 0 and journal[start].role == "assistant":
            start -= 1
        return start

    def _budget_lines(self, items: Sequence[MemoryItem], budget: int
                      ) -> Tuple[List[str], List[int]]:
        """Newest-first admission = LRU eviction of the oldest once the layer's
        token budget is spent (design §6); returned chronological."""
        lines: List[str] = []
        ids: List[int] = []
        used = 0
        for item in reversed(list(items)):
            text = inject_mod.sanitize_model_text(item.text).replace("\n", " ").strip()
            if not text:
                continue
            line = f"[{inject_mod.format_stamp(item.session_ts)}] {text}"
            cost = inject_mod.estimate_tokens(line)
            if used + cost > budget:
                continue  # keep scanning: a short older line may still fit
            lines.append(line)
            ids.append(item.id)
            used += cost
        lines.reverse()
        ids.reverse()
        return lines, ids

    def _verbatim_hold(self, journal: Sequence[MemoryItem]) -> Tuple[List[str], List[int]]:
        """Copied character-exact (the summarizer never sees these): user
        corrections, identifiers/proper nouns, numbers with units, assistant
        commitments. Lexical only — correction detection has no model (§7)."""
        picked: List[MemoryItem] = []
        for item in reversed(list(journal)):  # newest first → LRU eviction
            text = (item.text or "").strip()
            if not text:
                continue
            if item.role == "user" and _CORRECTION_RE.search(text):
                picked.append(item)
            elif item.role == "assistant" and _COMMITMENT_RE.search(text):
                picked.append(item)
            elif _NUMBER_UNIT_RE.search(text) or _IDENTIFIER_RE.search(text):
                picked.append(item)
        return self._budget_lines(list(reversed(picked)), _BUDGET_VERBATIM_HOLD)

    def _assemble(self, summary: Optional[str],
                  collected: Optional[Tuple[List[MemoryItem], List[MemoryItem], List[MemoryItem]]] = None,
                  pins: Optional[Sequence[str]] = None,
                  ) -> Tuple[List[dict], List[int], int]:
        """System message = base prompt + recall declaration + pinned +
        verbatim-hold + summary + handle line, in THAT order (design §6); then
        the verbatim tail as real user/assistant messages. The summary is never
        an assistant-role message (voice drift) — it lives inside the system
        layer like every other compacted block."""
        journal, pinned, tail = collected if collected is not None else self._collect()
        lang = self._lang()
        zh = lang.startswith("zh")
        sections = [inject_mod.augment_system_prompt(self._base_prompt(), lang)]
        kept: List[int] = []

        pinned_lines, pinned_ids = self._budget_lines(pinned, _BUDGET_PINNED)
        if pins:
            # Explicit user pins take priority. Generated pins share the same
            # bounded layer rather than bypassing the prefix budget.
            used = sum(inject_mod.estimate_tokens(line) for line in pinned_lines)
            for pin in pins:
                if not isinstance(pin, str):
                    continue
                line = inject_mod.sanitize_model_text(pin).replace("\n", " ").strip()
                cost = inject_mod.estimate_tokens(line)
                if line and used + cost <= _BUDGET_PINNED:
                    pinned_lines.append(line)
                    used += cost
        if pinned_lines:
            header = "置顶记忆 / Pinned:" if zh else "Pinned memories:"
            sections.append(header + "\n" + "\n".join(pinned_lines))
            kept.extend(pinned_ids)

        hold_lines, hold_ids = self._verbatim_hold(journal)
        if hold_lines:
            header = "逐字保留的记忆 / Verbatim:" if zh else "Verbatim memories (exact):"
            sections.append(header + "\n" + "\n".join(hold_lines))
            kept.extend(hold_ids)

        if summary:
            header = "此前对话摘要 / Summary:" if zh else "Conversation summary so far:"
            sections.append(header + "\n" + summary)

        max_ts = max((it.session_ts or 0.0) for it in journal) if journal else 0.0
        stamp = inject_mod.format_stamp(max_ts)
        sections.append(f"记忆库覆盖 t=0…{stamp}" if zh else f"Memory bank covers t=0…{stamp}")

        system_text = "\n\n".join(s for s in sections if s)
        messages: List[dict] = [{"role": "system", "content": system_text}]
        est = inject_mod.estimate_tokens(system_text) + _TAIL_MSG_OVERHEAD

        tail_items = list(tail)
        # tail budget: drop the OLDEST turns first, then re-assert the QA
        # boundary so a shrunk tail still starts on a user turn
        while tail_items and sum(
                inject_mod.estimate_tokens(inject_mod.sanitize_model_text(it.text))
                for it in tail_items) > _BUDGET_TAIL:
            tail_items.pop(0)
        while tail_items and tail_items[0].role == "assistant":
            tail_items.pop(0)

        for item in tail_items:
            text = inject_mod.sanitize_model_text(item.text)
            if not text:
                continue
            role = "user" if item.role == "user" else "assistant"
            messages.append({"role": role, "content": text})
            kept.append(item.id)
            est += inject_mod.estimate_tokens(text) + _TAIL_MSG_OVERHEAD
        return messages, kept, est

    # ------------------------------------------------------------------ summary

    def _summary_configured(self) -> bool:
        return (self.settings.memory_summary_provider or "") in ("offline", "pi")

    async def _summarize_full(self, journal: Sequence[MemoryItem]
                              ) -> Tuple[Optional[str], List[str]]:
        """(summary, pins) dispatch: pi_agent /compact or the offline plane.
        Both degrade to (None, []) → verbatim-tail-only, never an exception."""
        if (self.settings.memory_summary_provider or "") == "pi":
            return await asyncio.to_thread(self._summarize_pi, journal)
        return (await self._summarize(journal)), []

    def _journal_lines(self, journal: Sequence[MemoryItem]) -> List[str]:
        """Serialize the raw journal for pi_agent /compact: sanitized (this is
        user-authored text reaching an LLM), with the session's
        uncommitted (interrupted) turns appended as trailing context."""
        lines: List[str] = []
        for item in journal:
            text = inject_mod.sanitize_model_text(item.text).replace("\n", " ").strip()
            if not text:
                continue
            who = "用户" if item.role == "user" else "助手"
            line = f"{who}: {text}"
            lines.append(line)
        if self._journal_extra is not None:
            try:
                extra = self._journal_extra()
            except Exception:  # noqa: BLE001
                extra = []
            for role, text in extra:
                text = inject_mod.sanitize_model_text(text).replace("\n", " ").strip()
                if text:
                    who = "用户" if role == "user" else "助手"
                    lines.append(f"{who}: {text} [interrupted]")
        return lines

    def _summarize_pi(self, journal: Sequence[MemoryItem]) -> Tuple[Optional[str], List[str]]:
        """pi_agent /compact (board parity). Blocking — async callers go through
        to_thread, the prefetch thread calls it directly. Any failure or an
        empty summary returns (None, []) → the caller degrades to
        verbatim-tail-only, exactly like the offline branch."""
        lines = self._journal_lines(journal)
        if not lines:
            return None, []
        try:
            response = self.pi.compact(self.conversation_id, "\n".join(lines))
        except Exception as exc:  # noqa: BLE001
            log.warning("rollover pi compact failed: %s", exc)
            return None, []
        if response is None:
            return None, []
        if (not isinstance(response, dict) or not isinstance(response.get("summary"), str)
                or len(response["summary"]) > 200 or not isinstance(response.get("pins"), list)
                or len(response["pins"]) > 16
                or any(not isinstance(p, str) or not p.strip() or len(p) > 256
                       for p in response["pins"])):
            log.warning("pi_agent /compact returned invalid or oversized output; using verbatim tail")
            return None, []
        summary = inject_mod.sanitize_model_text(response["summary"]).strip()
        pins = [p for p in
                (inject_mod.sanitize_model_text(pin).replace("\n", " ").strip()
                 for pin in (response.get("pins") or [])) if p]
        if not summary:
            log.warning("pi_agent /compact returned an empty summary; using verbatim tail")
            return None, []
        return summary, pins

    # ------------------------------------------------------------------ compact prefetch

    @while_open(False)
    def maybe_prefetch_compact(self, text_tokens: Any) -> bool:
        """Start a background _collect + pi /compact once text tokens cross
        `memory_rollover_idle_tokens * memory_rollover_prefetch_ratio` (default
        60%), so the seconds-long pi call is usually already done when the
        rollover itself fires. pi provider only — the local offline plane stays
        synchronous. At most one in-flight/ready prefetch per session."""
        if (self.settings.memory_summary_provider or "") != "pi":
            return False
        try:
            tokens = float(text_tokens)
        except (TypeError, ValueError):
            return False
        floor = float(self.settings.memory_rollover_idle_tokens) * max(
            0.0, float(self.settings.memory_rollover_prefetch_ratio))
        if floor <= 0 or tokens < floor:
            return False
        with self._prefetch_lock:
            existing = self._prefetch
            # running: never double-start; ready: leave it for build_prefix
            if existing is not None and existing.status in ("running", "ready"):
                return False
            self._prefetch = _Prefetch()
            prefetch = self._prefetch
        thread = threading.Thread(
            target=self._prefetch_worker, args=(prefetch,), daemon=True,
            name=f"memory-prefetch-{self.conversation_id[-6:]}")
        thread.start()
        return True

    @while_open()
    def _prefetch_worker(self, prefetch: _Prefetch) -> None:
        try:
            collected = self._collect()
            summary, pins = self._summarize_pi(collected[0])
            result = (collected, summary, pins) if summary else None
            status = "ready" if result is not None else "error"
        except Exception as exc:  # noqa: BLE001
            log.warning("rollover compact prefetch failed for %s: %s", self.conversation_id, exc)
            result, status = None, "error"
        with self._prefetch_lock:
            # a consumed/replaced prefetch discards this result silently
            if self._prefetch is prefetch and prefetch.status == "running":
                prefetch.status = status
                prefetch.result = result
                prefetch.done.set()

    def seal(self) -> None:
        self.lifetime.seal()
        with self._prefetch_lock:
            if self._prefetch is not None:
                self._prefetch.done.set()
            self._prefetch = None

    def close(self) -> None:
        self.seal()
        self.lifetime.wait()

    def _take_prefetch(self) -> Optional[Tuple[Tuple[List[MemoryItem], List[MemoryItem], List[MemoryItem]],
                                               str, List[str]]]:
        """Consume a finished prefetch; a still-running one gets a bounded join
        (30s) instead of a duplicated synchronous /compact call. A failed or
        empty prefetch returns None → the synchronous path runs unchanged."""
        deadline = time.monotonic() + 30.0
        while True:
            with self._prefetch_lock:
                prefetch = self._prefetch
            if prefetch is None or prefetch.status != "running":
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not prefetch.done.wait(timeout=min(remaining, 0.2)):
                if time.monotonic() >= deadline:
                    break
        with self._prefetch_lock:
            prefetch = self._prefetch
            self._prefetch = None  # consumed either way; an error may retry later
        if prefetch is None or prefetch.status != "ready" or prefetch.result is None:
            return None
        log.info("rollover reusing prefetched compact for %s", self.conversation_id)
        return prefetch.result

    # ------------------------------------------------------------------ summary (offline plane)

    async def _summarize(self, journal: Sequence[MemoryItem]) -> Optional[str]:
        """Summarize all raw QA blocks, merging chunks within this rollover.
        Any provider/plane/generation
        problem returns None → verbatim-tail-only, never an exception."""
        if not self._summary_configured():
            return None
        plane = self.plane
        if plane is None:
            return None
        try:
            if not plane.is_loaded():
                return None
        except Exception:  # noqa: BLE001
            return None
        lines = self._journal_lines(journal)
        if not lines:
            return None
        blocks: List[str] = []
        for line in lines:
            if line.startswith("用户:") or not blocks:
                blocks.append(line)
            else:
                blocks[-1] += "\n" + line
        chunks: List[str] = []
        for block in blocks:
            if inject_mod.estimate_tokens(block) > 2500:
                log.warning("offline compact QA block exceeds budget; using verbatim tail")
                return None
            if chunks and inject_mod.estimate_tokens(chunks[-1] + "\n" + block) <= 2500:
                chunks[-1] += "\n" + block
            else:
                chunks.append(block)
        if len(chunks) > 32:
            log.warning("offline compact exceeds 32 chunks; using verbatim tail")
            return None
        text = ""
        try:
            async with self._semaphore:
                for chunk in chunks:
                    context = (f"Previous compact state:\n{text}\nNew conversation:\n" if text else "") + chunk
                    req = ChatRequest(
                        messages=[ChatMessage(role="user", content=_SUMMARY_PROMPT.format(journal=context))],
                        params=GenerationParams(
                            max_new_tokens=min(200, max(16, int(self.settings.memory_summary_max_tokens))),
                            temperature=0.0))
                    parts: List[str] = []
                    async for delta in plane.generate_stream(req):
                        parts.append(str(delta))
                    text = inject_mod.sanitize_model_text("".join(parts)).strip()
                    if not text or inject_mod.estimate_tokens(text) > 400:
                        log.warning("offline compact returned empty/oversized summary; using verbatim tail")
                        return None
        except Exception as exc:  # noqa: BLE001 — degrade to verbatim-tail-only
            log.warning("rollover summary failed: %s", exc)
            return None
        return text or None
