"""Check the MOSS-VL delivery environment without loading model weights."""

import argparse
import importlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORE_VERSIONS = {
    "torch": "2.11.0",
    "torchvision": "0.26.0",
    "torchaudio": "2.11.0",
    "transformers": "5.12.1",
    "sglang": "0.5.16",
    "flashinfer-python": "0.6.14",
    "torchcodec": "0.11.1",
}


def check_model(directory):
    directory = directory.expanduser().resolve()
    required = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
        "chat_template.json",
    ]
    for name in required:
        if not (directory / name).is_file():
            raise ValueError(f"Missing model file: {name}")
    config = json.loads((directory / "config.json").read_text())
    if not any("MossVL" in name for name in config.get("architectures", [])):
        raise ValueError("Expected a MOSS-VL checkpoint")
    for name in (
        "config.json",
        "tokenizer_config.json",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
        "processor_config.json",
    ):
        if not (directory / name).is_file():
            continue
        metadata = json.loads((directory / name).read_text())
        for mapping in metadata.get("auto_map", {}).values():
            for value in mapping if isinstance(mapping, list) else [mapping]:
                if not isinstance(value, str):
                    continue
                module = value.rsplit("--", 1)[-1].rsplit(".", 1)[0]
                module_file = directory / (module.replace(".", "/") + ".py")
                if not module_file.is_file():
                    raise ValueError(f"Missing custom model code: {module_file.name}")
    index = directory / "model.safetensors.index.json"
    if index.is_file():
        shards = set(json.loads(index.read_text())["weight_map"].values())
    else:
        shards = {"model.safetensors"}
    if not shards:
        raise ValueError("Empty model weight index")
    for name in shards:
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Missing or invalid weight shard: {name}")
        # Hub snapshots link shard names to blobs outside the snapshot directory.
        path = directory / relative
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing or invalid weight shard: {name}")
    return f"{directory} ({len(shards)} weight file(s); weights not loaded)"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model_path", type=Path, nargs="?", help="Optional downloaded model directory"
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Check packages/files without GPU runtime imports",
    )
    args = parser.parse_args(argv)
    errors = []

    def record(name, fn):
        try:
            detail = fn()
        except Exception as exc:
            errors.append(name)
            print(f"ERROR {name}: {exc}", flush=True)
        else:
            print(f"OK    {name}: {detail}", flush=True)

    def python_platform():
        if sys.version_info[:2] != (3, 12):
            raise ValueError("The delivery profile requires Python 3.12")
        if platform.system() != "Linux" or platform.machine() != "x86_64":
            raise ValueError("The pinned delivery profile targets Linux x86_64")
        return f"Python {platform.python_version()}, Linux x86_64"

    record("platform", python_platform)
    for name, expected in CORE_VERSIONS.items():

        def package(name=name, expected=expected):
            actual = importlib.metadata.version(name)
            if actual.partition("+")[0] != expected:
                raise ValueError(f"expected {expected}, installed {actual}")
            return actual

        record(name, package)
    record("nvidia-ml-py", lambda: importlib.metadata.version("nvidia-ml-py"))

    def checkout():
        module = importlib.import_module("sglang_omni")
        location = Path(module.__file__).resolve().parent
        if location != ROOT / "sglang_omni":
            raise ValueError(
                f"Imported another checkout: {location}; install this repository with -e ."
            )
        return str(location)

    record("checkout", checkout)
    if args.model_path is not None:
        record("model files", lambda: check_model(args.model_path))
    if not args.no_gpu and not errors:

        def runtime():
            torch = importlib.import_module("torch")
            if not torch.version.cuda or not torch.version.cuda.startswith("13."):
                raise ValueError(
                    f"Expected CUDA 13 PyTorch, found {torch.version.cuda}"
                )
            if not torch.cuda.is_available():
                raise ValueError(
                    "CUDA is unavailable; check driver and CUDA_VISIBLE_DEVICES"
                )
            probe = torch.ones(1, device="cuda")
            if (probe + 1).item() != 2:
                raise RuntimeError("CUDA tensor check failed")
            importlib.import_module("sglang_omni.models.moss_vl_realtime.stages")
            return f"{torch.cuda.get_device_name(0)}, CUDA {torch.version.cuda}, realtime imports ready"

        record("GPU runtime", runtime)
    elif args.no_gpu:
        print("SKIP  GPU runtime (--no-gpu)")
    if errors:
        print("FAIL: see docs/get_started/installation.md")
        return 1
    print(
        "PASS: environment checks complete; server startup warmup verifies model/JIT execution"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
