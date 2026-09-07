"""Check Qwen3's preparation fallback without requiring Ascend runtime imports."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


path = Path(__file__).resolve().parents[3] / "python/sglang/srt/models/qwen3.py"
tree = ast.parse(path.read_text())
attention = next(
    node
    for node in tree.body
    if isinstance(node, ast.ClassDef) and node.name == "Qwen3Attention"
)
method = next(
    node
    for node in attention.body
    if isinstance(node, ast.FunctionDef) and node.name == "forward_prepare_npu"
)
# Execute the real method, isolating hardware-only imports elsewhere in the model.
code = compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec")


class TestQwen3NpuFallback(unittest.TestCase):
    def test_missing_kernel_uses_native_preparation(self):
        namespace = {"split_qkv_rmsnorm_rope": None}
        exec(code, namespace)
        expected = (object(), object(), object())
        attention = SimpleNamespace(forward_prepare_native=Mock(return_value=expected))
        positions, hidden_states = object(), object()
        actual = namespace["forward_prepare_npu"](
            attention, positions, hidden_states, None
        )
        self.assertIs(actual, expected)
        attention.forward_prepare_native.assert_called_once_with(
            positions, hidden_states
        )

    def test_available_kernel_keeps_fused_preparation(self):
        expected = (object(), object(), object())
        kernel = Mock(return_value=expected)
        namespace = {"split_qkv_rmsnorm_rope": kernel}
        exec(code, namespace)
        attention = SimpleNamespace(
            forward_prepare_native=Mock(),
            qkv_proj=Mock(return_value=(object(), None)),
            attn=SimpleNamespace(layer_id=1),
            rotary_emb=SimpleNamespace(position_sin=None, position_cos=None),
            q_size=2,
            kv_size=1,
            head_dim=1,
            q_norm=SimpleNamespace(variance_epsilon=1e-6, weight=None),
            k_norm=SimpleNamespace(weight=None),
        )
        batch = SimpleNamespace(token_to_kv_pool=SimpleNamespace(start_layer=0))
        actual = namespace["forward_prepare_npu"](attention, None, None, batch)
        self.assertEqual(actual, expected)
        kernel.assert_called_once()
        attention.forward_prepare_native.assert_not_called()


if __name__ == "__main__":
    unittest.main()
