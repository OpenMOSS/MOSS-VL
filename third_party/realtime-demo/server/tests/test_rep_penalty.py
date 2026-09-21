"""Selective repetition penalty unit tests (no GPU, no models, stub tokenizer).

Run:  <repo>/.venv/bin/python -m server.tests.test_rep_penalty

Covers the safety properties of server/adapters/vlm/moss_vl_hf/rep_penalty.py:
exempt ids (above all <|silence|>) are never touched, CTRL sign semantics,
filter-then-window (scaffold splices must not consume the window), no-op edges,
clamping + fail-open-to-OFF, configure_generation wiring (stock neutralized,
nothing constructed at penalty=1.0), schema default None, and the lazy
device/vocab-dim exemption-mask rebuild.
"""
from __future__ import annotations

import sys

import torch

from server.adapters.vlm.moss_vl_hf.rep_penalty import (
    SelectiveRepetitionPenalty, build_exempt_ids, configure_generation)

# ---- stub tokenizers ---------------------------------------------------------

_ADDED = {
    "<|silence|>": 90, "<|response|>": 99, "<|im_start|>": 96, "<|im_end|>": 97,
    "<|endoftext|>": 98, "<|vision_start|>": 91, "<|vision_end|>": 92,
    "<|vision_pad|>": 89, "<|image_pad|>": 93, "<|video_pad|>": 88,
    "<|time_start|>": 94, "<|time_end|>": 95,
}
# single-token encode table (everything else encodes as two ids = excluded)
_SINGLE = {**{str(d): 10 + d for d in range(10)},
           ".": 3, ",": 4, " ": 5, "\n": 6, "!": 7, "?": 8,
           "。": 40, "，": 41, " seconds": 30, "seconds": 31}


class StubTok:
    def get_added_vocab(self):
        return dict(_ADDED)

    def convert_tokens_to_ids(self, token):
        return _ADDED.get(token)

    def convert_ids_to_tokens(self, ids):
        rev = {v: k for k, v in {**_ADDED, **_SINGLE}.items()}
        return [rev.get(i, f"tok{i}") for i in ids]

    def __call__(self, text, add_special_tokens=False):
        if text in _SINGLE:
            return {"input_ids": [_SINGLE[text]]}
        return {"input_ids": [201, 202]}  # multi-token → not exempt


class BrokenTok(StubTok):
    """Added-vocab APIs unavailable → literal-token fallback must engage."""

    def get_added_vocab(self):
        raise RuntimeError("no added vocab API")
    # no added_tokens_decoder / all_special_ids attributes → AttributeError


class NoSilenceTok(StubTok):
    def convert_tokens_to_ids(self, token):
        return None


def _scores(vocab=120):
    # deterministic mixed-sign scores: even ids positive, odd ids negative
    base = torch.arange(vocab, dtype=torch.float32)
    return torch.where(base % 2 == 0, base + 1.0, -(base + 1.0)).unsqueeze(0)


TEXT_A, TEXT_B, TEXT_C = 100, 101, 103  # plain-text ids (not exempt)


# ---- tests -------------------------------------------------------------------

def test_exempt_ids_and_scores_untouched() -> None:
    exempt = build_exempt_ids(StubTok())
    for tid in (90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 10, 19, 3, 40, 30, 5, 6):
        assert tid in exempt, f"id {tid} must be exempt"
    assert TEXT_A not in exempt and 201 not in exempt

    proc = SelectiveRepetitionPenalty(1.25, exempt, window=8)
    ids = torch.tensor([[90, 91, 10, 3, 30, TEXT_A, 90, TEXT_B]])
    before = _scores()
    after = proc(ids, before.clone())
    for tid in (90, 91, 10, 3, 30):
        assert torch.equal(after[0, tid], before[0, tid]), f"exempt id {tid} was modified"
    assert not torch.equal(after[0, TEXT_A], before[0, TEXT_A])
    assert not torch.equal(after[0, TEXT_B], before[0, TEXT_B])
    print("exempt untouched: OK")


def test_sign_semantics() -> None:
    proc = SelectiveRepetitionPenalty(1.25, [90], window=0)
    ids = torch.tensor([[TEXT_A, TEXT_B]])
    scores = torch.zeros(1, 120)
    scores[0, TEXT_A] = 2.0    # positive → divided
    scores[0, TEXT_B] = -2.0   # negative → multiplied (further down)
    scores[0, TEXT_C] = 4.0    # unseen → untouched
    out = proc(ids, scores.clone())
    assert torch.isclose(out[0, TEXT_A], torch.tensor(2.0 / 1.25))
    assert torch.isclose(out[0, TEXT_B], torch.tensor(-2.0 * 1.25))
    assert out[0, TEXT_C] == 4.0
    # -inf stays -inf
    scores[0, TEXT_A] = float("-inf")
    out = proc(ids, scores.clone())
    assert torch.isinf(out[0, TEXT_A]) and out[0, TEXT_A] < 0
    print("sign semantics: OK")


def test_filter_then_window() -> None:
    exempt = build_exempt_ids(StubTok())
    proc = SelectiveRepetitionPenalty(1.5, exempt, window=2)
    # tail = text_a, scaffold x5, text_b — a window-then-filter bug would only
    # see scaffold + text_b and miss text_a
    ids = torch.tensor([[TEXT_A, 91, 91, 91, 91, 91, TEXT_B]])
    before = _scores()
    after = proc(ids, before.clone())
    assert not torch.equal(after[0, TEXT_A], before[0, TEXT_A]), "text_a missed (window ate scaffold)"
    assert not torch.equal(after[0, TEXT_B], before[0, TEXT_B])
    # all-scaffold tail: the sole older text id must still be reachable
    ids = torch.tensor([[TEXT_A] + [91] * 20])
    after = proc(ids, before.clone())
    assert not torch.equal(after[0, TEXT_A], before[0, TEXT_A])
    print("filter-then-window: OK")


def test_noop_and_edges() -> None:
    exempt = build_exempt_ids(StubTok())
    before = _scores()
    # penalty == 1.0 → fast path, same object
    inert = SelectiveRepetitionPenalty(1.0, exempt, window=8)
    s = before.clone()
    assert inert(torch.tensor([[TEXT_A]]), s) is s
    # all-exempt context → unchanged values
    proc = SelectiveRepetitionPenalty(1.5, exempt, window=8)
    assert torch.equal(proc(torch.tensor([[90, 91, 10]]), before.clone()), before)
    # input_ids shorter than window / empty
    assert not torch.equal(proc(torch.tensor([[TEXT_A]]), before.clone()), before)
    empty = torch.empty(1, 0, dtype=torch.long)
    assert torch.equal(proc(empty, before.clone()), before)
    # window=0 → full context, no scan cap: ancient text still penalized
    full = SelectiveRepetitionPenalty(1.5, exempt, window=0)
    assert full.scan_cap == 0
    ids = torch.tensor([[TEXT_A] + [91] * 5000])
    assert not torch.equal(full(ids, before.clone()), before)
    # windowed scan cap bounds the reach
    assert SelectiveRepetitionPenalty(1.5, exempt, window=256).scan_cap == 4096
    print("no-op + edges: OK")


def test_clamping_and_failopen() -> None:
    assert SelectiveRepetitionPenalty(3.0, [90]).penalty == 2.0
    assert SelectiveRepetitionPenalty(0.5, [90]).penalty == 1.0  # inert floor
    # fail-open: silence unresolvable → penalty forced OFF, nothing injected
    gen = {"repetition_penalty": 1.2, "temperature": 0.7}
    out = configure_generation(gen, NoSilenceTok(), window=256, exempt=True)
    assert out["repetition_penalty"] == 1.0 and "logits_processor" not in out
    print("clamping + fail-open: OK")


def test_configure_generation_wiring() -> None:
    base = {"max_new_tokens": 512, "temperature": 0.7, "top_k": 20,
            "top_p": 0.8, "do_sample": True}
    # off (1.0): nothing constructed, dict unchanged
    gen = dict(base, repetition_penalty=1.0)
    out = configure_generation(gen, StubTok(), window=256, exempt=True)
    assert out == dict(base, repetition_penalty=1.0) and "logits_processor" not in out
    # None → normalized off
    gen = dict(base, repetition_penalty=None)
    out = configure_generation(gen, StubTok(), window=256, exempt=True)
    assert out["repetition_penalty"] == 1.0 and "logits_processor" not in out
    # selective: stock neutralized + exactly one processor injected
    gen = dict(base, repetition_penalty=1.1)
    out = configure_generation(gen, StubTok(), window=256, exempt=True)
    assert out["repetition_penalty"] == 1.0
    assert isinstance(out["logits_processor"], list) and len(out["logits_processor"]) == 1
    assert isinstance(out["logits_processor"][0], SelectiveRepetitionPenalty)
    # legacy passthrough: exempt off → stock penalty untouched, no injection
    gen = dict(base, repetition_penalty=1.15)
    out = configure_generation(gen, StubTok(), window=256, exempt=False)
    assert out == dict(base, repetition_penalty=1.15) and "logits_processor" not in out
    # broken added-vocab APIs → literal fallback still exempts silence
    gen = dict(base, repetition_penalty=1.1)
    out = configure_generation(gen, BrokenTok(), window=256, exempt=True)
    proc = out["logits_processor"][0]
    before = _scores()
    after = proc(torch.tensor([[90, TEXT_A]]), before.clone())
    assert torch.equal(after[0, 90], before[0, 90]), "silence penalized under fallback tokenizer"
    print("configure_generation wiring: OK")


def test_schema_default_none() -> None:
    from server.schemas import GenerationParams
    assert GenerationParams().repetition_penalty is None, \
        "realtime must fall back to Settings (GEN_REPETITION_PENALTY)"
    print("schema default None: OK")


def test_device_mask_rebuild() -> None:
    proc = SelectiveRepetitionPenalty(1.5, [90], window=8)
    before120 = _scores(120)
    out = proc(torch.tensor([[90, TEXT_A]]), before120.clone())
    assert torch.equal(out[0, 90], before120[0, 90])
    # smaller vocab dim: exempt id 90 >= 60 is dropped without error
    before60 = _scores(60)
    out = proc(torch.tensor([[33]]), before60.clone())
    assert not torch.equal(out[0, 33], before60[0, 33])
    print("device/vocab mask rebuild: OK")


def main() -> int:
    test_exempt_ids_and_scores_untouched()
    test_sign_semantics()
    test_filter_then_window()
    test_noop_and_edges()
    test_clamping_and_failopen()
    test_configure_generation_wiring()
    test_schema_default_none()
    test_device_mask_rebuild()
    print("\nREP PENALTY TEST OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
