"""Selective repetition penalty for the MOSS-VL realtime loop.

The stock HF ``RepetitionPenaltyLogitsProcessor`` penalizes every id present in
``input_ids`` — unsafe for this streaming checkpoint, whose context contains
control/scaffold tokens by construction: ``<|silence|>`` (the idle/turn-end
channel, model-emitted every idle step AND manually spliced by the loop),
per-frame vision/time scaffolding, turn scaffolding, and the per-frame
timestamp text (``"{ts:.1f} seconds"``). Penalizing silence suppresses the
model's ability to go idle (→ hallucinated narration rounds; cf.
VideoLLM-online's protected EOS channel, arXiv 2406.11816); penalizing
digits/punctuation starves natural repeaters (llama.cpp ``--no-penalize-nl``
precedent, ggml-org/llama.cpp#3675).

``SelectiveRepetitionPenalty`` is a CTRL-style multiplicative penalty over the
last-K **non-exempt** ids (filter-then-window — an idle stretch is ~100%
exempt scaffold, so window-then-filter would silently disable the guard
exactly when cross-round loops happen), with exemption tiers resolved from the
live tokenizer. Stateless over ``input_ids`` → immune to the frame-window
eviction patch (which trims cross-attn KV only, never input_ids).

Injected by the adapter as a caller-supplied ``logits_processor`` — the
checkpoint's ``_real_time_generate`` merges it via ``_get_logits_processor``
(modeling_moss_vl.py:3011) and applies it each step BEFORE the top-k/top-p
warpers (transformers 4.57.1 utils.py:1248 vs :1251 — llama.cpp order;
re-check on transformers upgrades). Torch-only on purpose: importable in the
worker process and in CPU unit tests; ``LogitsProcessorList`` duck-types on
``__call__(input_ids, scores)``, no subclassing needed.
"""
from __future__ import annotations

import string
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

from ....logging_conf import get_logger

log = get_logger(__name__)

# The idle/turn-end channel. If this id cannot be resolved, the penalty is
# disabled entirely (fail open to OFF — never to penalized-silence).
SILENCE_TOKEN = "<|silence|>"

# Literal fallback for tier (a) when the tokenizer exposes no added-vocab API.
# NOTE: <|round_start|>/<|eot_id|>/<|round_end|>/<|...|>/<|assistant|> are
# server-side queue signal strings, NOT vocab tokens — nothing to exempt.
CONTROL_TOKEN_LITERALS = (
    SILENCE_TOKEN, "<|response|>", "<|im_start|>", "<|im_end|>", "<|endoftext|>",
    "<|vision_start|>", "<|vision_end|>", "<|vision_pad|>",
    "<|image_pad|>", "<|video_pad|>", "<|time_start|>", "<|time_end|>",
)

# Tier (b): natural repeaters that must never be starved. ASCII digits +
# punctuation, CJK punctuation, space, newline.
_TEXT_EXEMPT_CANDIDATES = (
    tuple(string.digits)
    + tuple(string.punctuation)
    # CJK punctuation + full-width quotes (“”‘’ = curly quotes)
    + tuple("。，！？、；：（）《》【】…—·“”‘’")
    + (" ", "\n")
    # Tier (c): the per-frame timestamp literal from the realtime loop's
    # scaffold f"{ts:.1f} seconds" (modeling_moss_vl.py:2683) — recurs in
    # input_ids on every frame.
    + (" seconds", "seconds")
)


def _added_vocab_ids(tokenizer: Any) -> List[int]:
    """Tier (a): every added/control token id, best API first."""
    getters: Tuple[Callable[[], Any], ...] = (
        lambda: tokenizer.get_added_vocab().values(),
        lambda: tokenizer.added_tokens_decoder.keys(),
        lambda: tokenizer.all_special_ids,
    )
    for getter in getters:
        try:
            ids = [int(i) for i in getter() if i is not None and int(i) >= 0]
        except Exception:  # noqa: BLE001 — fall through to the next API
            continue
        if ids:
            return ids
    return []


def _single_token_id(tokenizer: Any, text: str) -> Optional[int]:
    """Id of `text` iff it round-trips as exactly one token (else None)."""
    try:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    except Exception:  # noqa: BLE001
        return None
    if isinstance(ids, (list, tuple)) and len(ids) == 1 and int(ids[0]) >= 0:
        return int(ids[0])
    return None


def build_exempt_ids(tokenizer: Any, include_text_exemptions: bool = True) -> List[int]:
    """Resolve the exemption set from the live tokenizer (no hardcoded ids).

    Raises RuntimeError when the silence token cannot be resolved — the caller
    must then disable the penalty (fail open to OFF).
    """
    silence_id = None
    try:
        silence_id = tokenizer.convert_tokens_to_ids(SILENCE_TOKEN)
    except Exception:  # noqa: BLE001
        pass
    if silence_id is None or int(silence_id) < 0:
        raise RuntimeError(
            f"cannot resolve {SILENCE_TOKEN} id — refusing to run a repetition "
            "penalty that might suppress the idle channel")

    exempt = {int(silence_id)}
    exempt.update(_added_vocab_ids(tokenizer))
    # literal control tokens: also covers tokenizers whose added-vocab APIs are
    # unavailable/partial (e.g. all_special_ids missing special=False entries)
    for token in CONTROL_TOKEN_LITERALS:
        try:
            tid = tokenizer.convert_tokens_to_ids(token)
        except Exception:  # noqa: BLE001
            continue
        if tid is not None and int(tid) >= 0:
            exempt.add(int(tid))
    if include_text_exemptions:
        for cand in _TEXT_EXEMPT_CANDIDATES:
            tid = _single_token_id(tokenizer, cand)
            if tid is not None:
                exempt.add(tid)
    return sorted(exempt)


class SelectiveRepetitionPenalty:
    """CTRL-style repetition penalty over the last-K non-exempt context ids.

    Duck-typed HF logits processor (``__call__(input_ids, scores) -> scores``).
    Only reads batch row 0 — the realtime loop asserts batch_size == 1
    (modeling_moss_vl.py:2965); extra rows would pass through untouched.
    """

    #: values outside this band are clamped hard
    PENALTY_MIN, PENALTY_MAX = 1.0, 2.0
    #: values outside this band draw a warning (aggressive penalties are a known
    #: cause of loops themselves — Unsloth/QwQ sampler findings)
    SANE_MAX = 1.3

    def __init__(self, penalty: float, exempt_ids: Sequence[int], window: int = 256):
        requested = float(penalty)
        self.penalty = min(max(requested, self.PENALTY_MIN), self.PENALTY_MAX)
        if requested != self.penalty or requested > self.SANE_MAX:
            log.warning(
                "repetition penalty %.3f outside the sane band [%.1f, %.1f] — using %.3f",
                requested, self.PENALTY_MIN, self.SANE_MAX, self.penalty)
        self.window = int(window)
        # raw-scan cap: bounds per-step cost AND time-localizes the window on an
        # endless stream (~40 tok/s context growth → last few minutes). Full-
        # context mode (window<=0) disables the cap by design.
        self.scan_cap = 0 if self.window <= 0 else max(4096, 16 * self.window)
        self._exempt_cpu = torch.tensor(sorted(set(int(i) for i in exempt_ids)),
                                        dtype=torch.long)
        # lazy per-(device, vocab-dim) exemption mask; sized from the LOGITS dim
        # (not len(tokenizer)) so gathers stay in-range even for untrained rows
        self._mask: Optional[torch.Tensor] = None
        self._mask_key: Optional[Tuple[Any, int]] = None
        log.info("SelectiveRepetitionPenalty: penalty=%.3f window=%d scan_cap=%d exempt_n=%d",
                 self.penalty, self.window, self.scan_cap, self._exempt_cpu.numel())

    def _exempt_mask(self, device: torch.device, vocab_dim: int) -> torch.Tensor:
        key = (device, vocab_dim)
        if self._mask is None or self._mask_key != key:
            mask = torch.zeros(vocab_dim, dtype=torch.bool, device=device)
            valid = self._exempt_cpu[self._exempt_cpu < vocab_dim].to(device)
            if valid.numel():
                mask[valid] = True
            self._mask, self._mask_key = mask, key
        return self._mask

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.penalty == 1.0 or input_ids.numel() == 0:
            return scores
        ids = input_ids[0]
        if self.scan_cap > 0:
            ids = ids[-self.scan_cap:]
        mask = self._exempt_mask(scores.device, scores.shape[-1])
        kept = ids[~mask[ids]]                       # filter FIRST …
        if self.window > 0:
            kept = kept[-self.window:]               # … then window
        if kept.numel() == 0:
            return scores
        uniq = torch.unique(kept)
        sel = scores[0, uniq]
        # CTRL semantics: push seen tokens down on both sides of zero;
        # -inf stays -inf under either branch (penalty > 0)
        penalized = torch.where(sel < 0, sel * self.penalty, sel / self.penalty)
        return scores.scatter(1, uniq.unsqueeze(0), penalized.unsqueeze(0))


def configure_generation(gen: Dict[str, Any], tokenizer: Any, *, window: int,
                         exempt: bool, wrap: Callable[[list], Any] = list) -> Dict[str, Any]:
    """Rewrite the realtime `gen` kwargs for the selective penalty.

    penalty == 1.0 (or None)  → OFF: no processor constructed, gen unchanged.
    exempt == False           → legacy stock HF full-context penalty (escape
                                hatch; penalizes silence — debugging only).
    else                      → stock processor neutralized (repetition_penalty
                                → 1.0) + one SelectiveRepetitionPenalty injected
                                via `wrap` (the adapter passes
                                transformers.LogitsProcessorList; tests pass
                                plain `list`).
    """
    rp = gen.get("repetition_penalty")
    rp = 1.0 if rp is None else float(rp)
    gen["repetition_penalty"] = rp
    if rp == 1.0:
        log.info("repetition penalty: off")
        return gen
    if not exempt:
        log.info("repetition penalty: stock HF full-context, penalty=%.3f "
                 "(GEN_REP_PENALTY_EXEMPT=0 — penalizes <|silence|>; debugging only)", rp)
        return gen
    try:
        exempt_ids = build_exempt_ids(tokenizer)
    except Exception as exc:  # noqa: BLE001 — fail OPEN to off, never to penalized-silence
        log.error("repetition penalty DISABLED (fail-open): %s", exc)
        gen["repetition_penalty"] = 1.0
        return gen
    proc = SelectiveRepetitionPenalty(rp, exempt_ids, window=window)
    gen["repetition_penalty"] = 1.0  # neutralize the stock processor (utils.py:1141)
    gen["logits_processor"] = wrap([proc])
    samples: List[str] = []
    try:
        head = exempt_ids[:4] + exempt_ids[-4:]
        samples = [str(t) for t in tokenizer.convert_ids_to_tokens(head)]
    except Exception:  # noqa: BLE001 — sample decode is best-effort logging only
        pass
    log.info("repetition penalty: selective, penalty=%.3f window=%d exempt_n=%d sample=%s",
             proc.penalty, proc.window, len(exempt_ids), samples)
    return gen
