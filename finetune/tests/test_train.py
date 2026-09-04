import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


finetune_dir = Path(__file__).resolve().parents[1]

torch = types.ModuleType("torch")
torch.Tensor = object()
torch.bfloat16 = object()
torch.cuda = SimpleNamespace(synchronize=MagicMock())

transformers = types.ModuleType("transformers")
transformers.TrainingArguments = object
transformers.AutoModelForCausalLM = object()
transformers.AutoProcessor = object()
transformers.Trainer = object()
transformers.HfArgumentParser = object()

arguments_path = finetune_dir / "arguments.py"
arguments_spec = importlib.util.spec_from_file_location("arguments", arguments_path)
arguments = importlib.util.module_from_spec(arguments_spec)

data = types.ModuleType("data")
data.MossVLSFTDataset = object()
data.MossVLDataCollator = object()

with patch.dict(
    sys.modules,
    {
        "torch": torch,
        "transformers": transformers,
        "arguments": arguments,
        "data": data,
    },
):
    arguments_spec.loader.exec_module(arguments)
    train_path = finetune_dir / "train.py"
    train_spec = importlib.util.spec_from_file_location("moss_vl_train", train_path)
    train = importlib.util.module_from_spec(train_spec)
    sys.modules["moss_vl_train"] = train
    train_spec.loader.exec_module(train)


class StopAfterModelLoad(Exception):
    pass


class TrainModelLoadTest(unittest.TestCase):
    def run_until_model_load(self, cross_attention_implementation):
        model_args = arguments.ModelArguments(
            model_name_or_path="checkpoint",
            cross_attention_implementation=cross_attention_implementation,
        )
        parser = MagicMock()
        parser.parse_args_into_dataclasses.return_value = (
            model_args,
            SimpleNamespace(),
            SimpleNamespace(local_rank=-1),
        )
        processor = MagicMock()
        model_loader = MagicMock()
        model_loader.from_pretrained.side_effect = StopAfterModelLoad

        with (
            patch.object(train.transformers, "HfArgumentParser", return_value=parser),
            patch.object(train, "AutoProcessor") as processor_loader,
            patch.object(train, "AutoModelForCausalLM", model_loader),
            self.assertRaises(StopAfterModelLoad),
        ):
            processor_loader.from_pretrained.return_value = processor
            train.train()

        return model_loader.from_pretrained.call_args.kwargs

    def test_default_does_not_pass_cross_attention_override(self):
        kwargs = self.run_until_model_load(None)

        self.assertEqual(kwargs["attn_implementation"], "flash_attention_2")
        self.assertNotIn("cross_attention_implementation", kwargs)

    def test_explicit_cross_attention_override_is_forwarded(self):
        kwargs = self.run_until_model_load("flash_attention_3")

        self.assertEqual(
            kwargs["cross_attention_implementation"],
            "flash_attention_3",
        )


if __name__ == "__main__":
    unittest.main()
