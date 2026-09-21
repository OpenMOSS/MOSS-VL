"""Frame-window eviction test: keep the last N frames' vision KV, keep the words.

Exercises `install_realtime_frame_window_patch` against a fake cross-attention KV
cache + synthetic `full_vision_token_info`, with NO GPU / no model load. Verifies:
  - only the cross-attention layers are trimmed (contiguous front drop); text
    (self-attention) layers are untouched;
  - `full_vision_token_info` offsets are shifted gap-free and totals decremented;
  - the rebuilt frame-level `cross_attention_mask` uses the evicted-frame offset so
    a text token attends to exactly the retained frames it has seen;
  - the window is a no-op while under budget, and disabling (keep=0) reinstalls clean.

Run:  <repo>/.venv/bin/python -m server.tests.test_frame_window
"""
from __future__ import annotations

import sys
from types import SimpleNamespace

import torch

from server.realtime.mossvl_patches import install_realtime_frame_window_patch

VISION_TOKENS_PER_FRAME = 4          # per frame, before the +1 separator
FRAME_LEN = VISION_TOKENS_PER_FRAME + 1  # media length in the cross-attn cache
IMAGE_TOKEN_ID = 151655
CROSS_LAYERS = [1, 3]                 # layer 0/2 = self-attn (text), 1/3 = cross (vision)
N_LAYERS = 4


class FakeLayer:
    def __init__(self, seq_len: int, tag: float):
        # [batch, heads, seq, head_dim]; fill with the position index so we can prove
        # WHICH tokens survived a slice, plus a per-layer tag to prove identity.
        base = torch.arange(seq_len, dtype=torch.float32).view(1, 1, seq_len, 1)
        self.keys = base.expand(1, 2, seq_len, 3).clone() + tag
        self.values = self.keys.clone()


class FakeCache:
    def __init__(self, text_len: int, vision_len: int):
        self.layers = []
        for i in range(N_LAYERS):
            seq = vision_len if i in CROSS_LAYERS else text_len
            self.layers.append(FakeLayer(seq, tag=100.0 * i))


def _make_model(keep_minutes_frames: int):
    model = SimpleNamespace()
    # Mirror the REAL MOSS-VL config layout: cross_attention_layers is nested
    # under text_config (NOT top-level), image_token_id stays top-level. The
    # original top-level-only lookup passed this mock but broke on the real
    # checkpoint (500 at session create) — keep this nested so it can't regress.
    model.config = SimpleNamespace(
        text_config=SimpleNamespace(cross_attention_layers=list(CROSS_LAYERS)),
        image_token_id=IMAGE_TOKEN_ID,
    )
    # Stub the "original" per-step hook: the real one has already appended tokens and
    # rebuilt state; here it just passes (input_ids, model_kwargs) through unchanged.
    model._update_model_kwargs_for_real_time_generation = lambda outputs, input_ids, mk, *a, **k: (input_ids, mk)
    install_realtime_frame_window_patch(model, keep_minutes_frames)
    return model


def _synthetic(num_frames: int):
    """Build input_ids (one <|image_pad|> per frame, text between) + full_vision_token_info + cache."""
    medias = [{
        "start": i * FRAME_LEN, "end": i * FRAME_LEN + FRAME_LEN, "length": FRAME_LEN,
        "num_frames": 1, "grid_h": 4, "grid_w": 2, "vision_tokens_per_frame": VISION_TOKENS_PER_FRAME,
        "has_separator": True,
    } for i in range(num_frames)]
    total = num_frames * FRAME_LEN
    fvti = [{"medias": medias, "total_length": total, "pad_start": total, "pad_end": total}]
    # input_ids: [txt, PAD, txt, PAD, ...] — one pad per frame, a trailing text token.
    ids = []
    for _ in range(num_frames):
        ids += [42, IMAGE_TOKEN_ID]
    ids += [42]
    input_ids = torch.tensor([ids], dtype=torch.long)
    cache = FakeCache(text_len=len(ids), vision_len=total)
    mk = {"full_vision_token_info": fvti, "past_key_values": cache,
          "cross_attention_mask": torch.zeros(1, 1, len(ids), num_frames, dtype=torch.bool)}
    return input_ids, mk, cache


def test_evicts_old_frames_keeps_words() -> None:
    keep = 3
    model = _make_model(keep)
    input_ids, mk, cache = _synthetic(num_frames=8)
    text_len_before = cache.layers[0].keys.shape[-2]

    input_ids, mk = model._update_model_kwargs_for_real_time_generation(None, input_ids, mk, True)

    # cross layers trimmed to keep*FRAME_LEN; self layers untouched
    for i in range(N_LAYERS):
        seq = cache.layers[i].keys.shape[-2]
        if i in CROSS_LAYERS:
            assert seq == keep * FRAME_LEN, (i, seq)
            # surviving tokens are the TAIL (positions 25..39 kept for an 8-frame cache)
            first_kept_pos = cache.layers[i].keys[0, 0, 0, 0].item() - 100.0 * i
            assert first_kept_pos == (8 - keep) * FRAME_LEN, first_kept_pos
        else:
            assert seq == text_len_before, (i, seq)  # WORDS untouched

    # full_vision_token_info: 3 kept, gap-free (first starts at 0), totals decremented
    fvti = mk["full_vision_token_info"][0]
    assert len(fvti["medias"]) == keep
    assert fvti["medias"][0]["start"] == 0 and fvti["medias"][0]["end"] == FRAME_LEN
    assert fvti["total_length"] == keep * FRAME_LEN == fvti["pad_end"]
    assert model._frame_window_evicted_total == 8 - keep

    # mask: (1,1,T,keep). The final text token has seen all 8 pads → attends to all 3 kept.
    mask = mk["cross_attention_mask"]
    assert mask.shape == (1, 1, input_ids.shape[1], keep), mask.shape
    assert bool((~mask[0, 0, -1]).all()), "last token must SEE all retained frames (mask False=visible)"
    # a token right after the 6th pad (overall frame idx 5 = kept idx 0) sees only kept frame 0
    sixth_pad_col = 2 * 6 - 1  # position of the 6th <|image_pad|>
    row = ~mask[0, 0, sixth_pad_col]  # True where visible
    assert bool(row[0]) and not bool(row[1]) and not bool(row[2]), row.tolist()
    print("evict old frames / keep words: OK")


def test_noop_under_budget() -> None:
    model = _make_model(10)
    input_ids, mk, cache = _synthetic(num_frames=4)  # 4 < 10 → nothing evicted
    input_ids, mk = model._update_model_kwargs_for_real_time_generation(None, input_ids, mk, True)
    for i in CROSS_LAYERS:
        assert cache.layers[i].keys.shape[-2] == 4 * FRAME_LEN
    assert model._frame_window_evicted_total == 0
    assert len(mk["full_vision_token_info"][0]["medias"]) == 4
    print("no-op under budget: OK")


def test_disable_after_enable() -> None:
    # A reused model instance: refreshing the budget to 0 disables eviction.
    model = _make_model(3)
    install_realtime_frame_window_patch(model, 0)  # already installed → refresh to off
    model._frame_window_evicted_total = 0
    input_ids, mk, cache = _synthetic(num_frames=8)
    input_ids, mk = model._update_model_kwargs_for_real_time_generation(None, input_ids, mk, True)
    for i in CROSS_LAYERS:
        assert cache.layers[i].keys.shape[-2] == 8 * FRAME_LEN, "disabled window must not evict"
    assert model._frame_window_evicted_total == 0
    print("disable after enable: OK")


def main() -> int:
    test_evicts_old_frames_keeps_words()
    test_noop_under_budget()
    test_disable_after_enable()
    print("\nFRAME WINDOW TEST OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
