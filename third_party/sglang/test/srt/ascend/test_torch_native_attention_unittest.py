"""CPU numerical regressions for the Ascend PyTorch attention implementation."""

import importlib.util
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F


path = (
    Path(__file__).resolve().parents[3]
    / "python/sglang/srt/hardware_backend/npu/attention/ascend_torch_native_backend.py"
)
spec = importlib.util.spec_from_file_location("ascend_torch_native_backend", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TestTorchNativeAttention(unittest.TestCase):
    def setUp(self):
        self.backend = module.AscendTorchNativeAttnBackend()
        torch.manual_seed(19)

    def test_softcap_mask_broadcast_and_empty_rows(self):
        q = torch.randn(2, 4, 3, 8)
        k, v = torch.randn(2, 2, 5, 8), torch.randn(2, 2, 5, 8)
        visible = torch.tensor(
            [[False] * 5, [True, False, True, False, False], [True] * 5]
        )
        for shape in ((3, 5), (1, 1, 3, 5)):
            for additive in (False, True):
                with self.subTest(shape=shape, additive=additive):
                    mask = visible.reshape(shape)
                    if additive:
                        mask = torch.zeros(shape).masked_fill(~mask, float("-inf"))
                    # With no cap, the helper must agree with SDPA's mask semantics.
                    actual = self.backend.scaled_dot_product_attention_with_softcapping(
                        q, k, v, attn_mask=mask, enable_gqa=True
                    )
                    expected = F.scaled_dot_product_attention(
                        q, k, v, attn_mask=mask, enable_gqa=True
                    )
                    torch.testing.assert_close(actual, expected)

                    actual = self.backend.scaled_dot_product_attention_with_softcapping(
                        q, k, v, attn_mask=mask, enable_gqa=True, logit_cap=1.5
                    )
                    repeated_k = k.repeat_interleave(2, dim=1)
                    repeated_v = v.repeat_interleave(2, dim=1)
                    expected = torch.zeros_like(q)
                    # Independent reference: compute only over each row's visible keys.
                    for row in (1, 2):
                        logits = (
                            q[:, :, row : row + 1]
                            @ repeated_k[:, :, visible[row]].transpose(-1, -2)
                        ) / (8**0.5)
                        weights = (1.5 * torch.tanh(logits / 1.5)).softmax(dim=-1)
                        expected[:, :, row : row + 1] = (
                            weights @ repeated_v[:, :, visible[row]]
                        )
                    torch.testing.assert_close(actual, expected)
                    self.assertTrue(torch.isfinite(actual).all())

    def test_packed_cross_attention_with_cached_prefix_and_mixed_batch(self):
        q = torch.zeros(6, 2, 2)
        k = torch.zeros(12, 1, 2)
        v = torch.arange(12, dtype=torch.float32)[:, None, None].expand(-1, 1, 2)
        lengths = torch.tensor([2, 1, 3])
        encoder_lens = torch.tensor([2, 0, 3])
        mask = torch.tensor([1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0], dtype=torch.uint8)
        expected = torch.tensor([0, 0, 0, 9, 9, 0], dtype=torch.float32)[
            :, None, None
        ].expand_as(q)
        for prefix in (torch.tensor([0, 0, 0]), torch.tensor([3, 2, 4])):
            for cap in (0.0, 1.5):
                with self.subTest(prefix=prefix.tolist(), cap=cap):
                    actual = self.backend.run_sdpa_forward_extend(
                        q,
                        torch.empty_like(q),
                        k,
                        v,
                        torch.arange(12).reshape(3, 4),
                        torch.arange(3),
                        prefix + lengths,
                        prefix,
                        lengths,
                        encoder_lens,
                        is_cross_attention=True,
                        enable_gqa=True,
                        logit_cap=cap,
                        cross_attention_custom_mask=mask,
                    )
                    torch.testing.assert_close(actual, expected)

    def test_causal_softcap_without_mask_matches_sdpa(self):
        q, k, v = (torch.randn(1, 2, 3, 4) for _ in range(3))
        actual = self.backend.scaled_dot_product_attention_with_softcapping(
            q, k, v, is_causal=True
        )
        torch.testing.assert_close(
            actual, F.scaled_dot_product_attention(q, k, v, is_causal=True)
        )


if __name__ == "__main__":
    unittest.main()
