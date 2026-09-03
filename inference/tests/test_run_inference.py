import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


torch = types.ModuleType("torch")
torch.bfloat16 = object()
transformers = types.ModuleType("transformers")
transformers.AutoModelForCausalLM = object()
transformers.AutoProcessor = object()

module_path = Path(__file__).resolve().parents[1] / "run_inference.py"
module_spec = importlib.util.spec_from_file_location("run_inference", module_path)
run_inference = importlib.util.module_from_spec(module_spec)
with patch.dict(sys.modules, {"torch": torch, "transformers": transformers}):
    module_spec.loader.exec_module(run_inference)
resolve_query_media_paths = run_inference.resolve_query_media_paths


class LoadModelTest(unittest.TestCase):
    def test_default_does_not_pass_cross_attention_override(self):
        processor_loader = MagicMock()
        model_loader = MagicMock()

        with (
            patch.object(run_inference, "AutoProcessor", processor_loader),
            patch.object(run_inference, "AutoModelForCausalLM", model_loader),
        ):
            run_inference.load_model("checkpoint")

        kwargs = model_loader.from_pretrained.call_args.kwargs
        self.assertEqual(kwargs["attn_implementation"], "flash_attention_2")
        self.assertNotIn("cross_attention_implementation", kwargs)

    def test_explicit_cross_attention_override_is_forwarded(self):
        processor_loader = MagicMock()
        model_loader = MagicMock()

        with (
            patch.object(run_inference, "AutoProcessor", processor_loader),
            patch.object(run_inference, "AutoModelForCausalLM", model_loader),
        ):
            run_inference.load_model(
                "checkpoint",
                cross_attention_implementation="flash_attention_3",
            )

        kwargs = model_loader.from_pretrained.call_args.kwargs
        self.assertEqual(
            kwargs["cross_attention_implementation"],
            "flash_attention_3",
        )


class ResolveQueryMediaPathsTest(unittest.TestCase):
    def test_normalizes_image_url_content_type_for_model(self):
        image_url = "https://example.com/images/sample.jpg"
        query = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": image_url},
                    ],
                }
            ],
        }

        resolved = resolve_query_media_paths(query, Path("/tmp/queries"))

        self.assertEqual(
            resolved["messages"][0]["content"][0],
            {"type": "image", "image_url": image_url},
        )

    def test_preserves_remote_image_references(self):
        image_url = "https://example.com/images/sample.jpg?size=large"
        query = {
            "images": [image_url],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_url},
                        {"type": "image", "image_url": image_url},
                    ],
                }
            ],
        }

        resolved = resolve_query_media_paths(query, Path("/tmp/queries"))

        self.assertEqual(resolved["images"], [image_url])
        self.assertEqual(resolved["messages"][0]["content"][0]["image"], image_url)
        self.assertEqual(
            resolved["messages"][0]["content"][1]["image_url"], image_url
        )

    def test_resolves_local_image_references(self):
        base_dir = Path("/tmp/queries")
        absolute_image = "/data/images/absolute.jpg"
        query = {
            "images": ["images/relative.jpg", absolute_image],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "images/inline.jpg"},
                        {"type": "image", "image_url": "images/url-field.jpg"},
                    ],
                }
            ],
        }

        resolved = resolve_query_media_paths(query, base_dir)

        self.assertEqual(
            resolved["images"],
            [str((base_dir / "images/relative.jpg").resolve()), absolute_image],
        )
        self.assertEqual(
            resolved["messages"][0]["content"][0]["image"],
            str((base_dir / "images/inline.jpg").resolve()),
        )
        self.assertEqual(
            resolved["messages"][0]["content"][1]["image_url"],
            str((base_dir / "images/url-field.jpg").resolve()),
        )


if __name__ == "__main__":
    unittest.main()
