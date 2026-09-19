import builtins
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


class DeviceUtilsTest(unittest.TestCase):
    def test_backend_selection_without_injecting_flash_attention(self):
        path = Path(__file__).resolve().parents[2] / "device_utils.py"
        original_import = builtins.__import__

        def import_without_flash(name, *args, **kwargs):
            if name == "flash_attn":
                raise ImportError("flash_attn is not installed")
            return original_import(name, *args, **kwargs)

        for npu, cuda, expected in [
            (True, False, "eager"),
            (False, True, "flash_attention_2"),
            (False, False, "eager"),
        ]:
            with self.subTest(npu=npu, cuda=cuda):
                torch = types.ModuleType("torch")
                torch.cuda = types.SimpleNamespace(is_available=lambda: cuda)
                torch.npu = types.SimpleNamespace(is_available=lambda: npu)
                spec = importlib.util.spec_from_file_location(
                    "tested_device_utils", path
                )
                module = importlib.util.module_from_spec(spec)
                with (
                    patch.dict(
                        sys.modules,
                        {
                            "torch": torch,
                            "torch_npu": types.ModuleType("torch_npu") if npu else None,
                        },
                    ),
                    patch("builtins.__import__", side_effect=import_without_flash),
                ):
                    sys.modules.pop("flash_attn", None)
                    spec.loader.exec_module(module)
                    self.assertEqual(module.resolve_attn_impl("auto"), expected)
                    self.assertNotIn("flash_attn", sys.modules)


if __name__ == "__main__":
    unittest.main()
