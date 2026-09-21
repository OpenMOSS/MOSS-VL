from __future__ import annotations

import argparse
import json
import hashlib
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import onnx
from onnx import TensorProto, external_data_helper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a MOSS-TTS-Nano Hugging Face checkpoint to TTS-only ONNX artifacts."
    )
    parser.add_argument(
        "--checkpoint-path",
        required=True,
        help="Path to the local Hugging Face MOSS-TTS-Nano checkpoint directory.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Path to the directory that will receive the exported ONNX artifacts.",
    )
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--sample-seq-len", type=int, default=24)
    parser.add_argument("--sample-past-len", type=int, default=24)
    parser.add_argument("--disable-eager-attn", action="store_true")
    return parser.parse_args()


def run_python_script(script_name: str, *extra_args: str) -> None:
    script_path = Path(__file__).resolve().parent / script_name
    command = [sys.executable, str(script_path), *extra_args]
    subprocess.run(command, check=True)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def externalize_onnx_file(onnx_path: Path) -> tuple[Path, Path]:
    resolved_path = onnx_path.expanduser().resolve()
    if resolved_path.suffix.lower() != ".onnx":
        raise ValueError(f"expected a .onnx file, got: {resolved_path}")
    data_path = resolved_path.with_suffix(".data")
    temp_onnx_path = resolved_path.with_suffix(".onnx.tmp")
    model = onnx.load_model(str(resolved_path), load_external_data=True)
    if data_path.exists():
        data_path.unlink()
    if temp_onnx_path.exists():
        temp_onnx_path.unlink()
    external_data_helper.convert_model_to_external_data(
        model,
        all_tensors_to_one_file=True,
        location=data_path.name,
        size_threshold=1024,
        convert_attribute=False,
    )
    onnx.save_model(model, str(temp_onnx_path))
    temp_onnx_path.replace(resolved_path)
    return resolved_path, data_path


def externalize_onnx_dir(model_dir: Path) -> None:
    for onnx_path in sorted(model_dir.glob("*.onnx")):
        externalize_onnx_file(onnx_path)


@dataclass(frozen=True)
class TensorBlob:
    digest: str
    data: bytes


def _is_external_tensor(tensor: TensorProto) -> bool:
    return tensor.data_location == TensorProto.EXTERNAL or bool(tensor.external_data)


def _tensor_blob(tensor: TensorProto) -> TensorBlob:
    raw = bytes(tensor.raw_data)
    if not raw:
        raise ValueError(f"tensor has no raw_data after load_external_data=True: {tensor.name}")
    return TensorBlob(
        digest=hashlib.sha256(raw).hexdigest(),
        data=raw,
    )


def merge_shared_external_data(onnx_paths: list[Path], shared_data_path: Path) -> None:
    resolved_paths = [path.expanduser().resolve() for path in onnx_paths]
    if not resolved_paths:
        raise ValueError("onnx_paths must not be empty")
    shared_data_path = shared_data_path.expanduser().resolve()
    models = [
        (
            path,
            onnx.load_model(str(path), load_external_data=False),
            onnx.load_model(str(path), load_external_data=True),
        )
        for path in resolved_paths
    ]
    unique_blobs: dict[str, tuple[int, int, bytes]] = {}
    shared_data_path.parent.mkdir(parents=True, exist_ok=True)
    file_bytes = bytearray()

    for _model_path, model_meta, model_data in models:
        for tensor_meta, tensor_data in zip(model_meta.graph.initializer, model_data.graph.initializer, strict=True):
            if not _is_external_tensor(tensor_meta):
                continue
            blob = _tensor_blob(tensor_data)
            if blob.digest in unique_blobs:
                continue
            offset = len(file_bytes)
            file_bytes.extend(blob.data)
            unique_blobs[blob.digest] = (offset, len(blob.data), blob.data)

    if shared_data_path.exists():
        shared_data_path.unlink()
    shared_data_path.write_bytes(file_bytes)

    for model_path, model_meta, model_data in models:
        for tensor_meta, tensor_data in zip(model_meta.graph.initializer, model_data.graph.initializer, strict=True):
            if not _is_external_tensor(tensor_meta):
                continue
            blob = _tensor_blob(tensor_data)
            offset, length, _raw = unique_blobs[blob.digest]
            tensor_meta.raw_data = blob.data
            tensor_meta.data_location = TensorProto.EXTERNAL
            external_data_helper.set_external_data(
                tensor_meta,
                location=shared_data_path.name,
                offset=offset,
                length=length,
            )
            tensor_meta.ClearField("raw_data")
        model_path.write_bytes(model_meta.SerializeToString())


def patch_tts_external_data_files(tts_meta_path: Path) -> None:
    payload: dict[str, Any] = json.loads(tts_meta_path.read_text(encoding="utf-8"))
    files = payload["files"]
    external_data_files = {
        files["prefill"]: ["moss_tts_global_shared.data"],
        files["decode_step"]: ["moss_tts_global_shared.data"],
        files["local_decoder"]: ["moss_tts_local_shared.data"],
        files["local_cached_step"]: ["moss_tts_local_shared.data"],
        files["local_fixed_sampled_frame"]: ["moss_tts_local_shared.data"],
    }
    ordered_payload = {
        "format_version": payload["format_version"],
        "checkpoint_path": payload["checkpoint_path"],
        "files": files,
        "external_data_files": external_data_files,
        "model_config": payload["model_config"],
        "onnx": payload["onnx"],
    }
    tts_meta_path.write_text(json.dumps(ordered_payload, indent=2, ensure_ascii=False), encoding="utf-8")


def cleanup_redundant_external_data(output_dir: Path) -> None:
    for path_value in [
        output_dir / "moss_tts_prefill.data",
        output_dir / "moss_tts_decode_step.data",
        output_dir / "moss_tts_local_cached_step.data",
        output_dir / "moss_tts_local_decoder.data",
        output_dir / "moss_tts_local_fixed_sampled_frame.data",
    ]:
        if path_value.exists():
            path_value.unlink()


def copy_tokenizer_model(checkpoint_path: Path, output_dir: Path) -> None:
    tokenizer_model_path = checkpoint_path / "tokenizer.model"
    if not tokenizer_model_path.exists():
        raise FileNotFoundError(f"tokenizer.model was not found under checkpoint path: {tokenizer_model_path}")
    shutil.copy2(tokenizer_model_path, output_dir / "tokenizer.model")


def export_tts_onnx(args: argparse.Namespace, output_dir: Path) -> None:
    checkpoint_path = Path(args.checkpoint_path).expanduser().resolve()
    run_python_script(
        "export_moss_tts_browser_onnx.py",
        "--checkpoint-path",
        str(checkpoint_path),
        "--output-dir",
        str(output_dir),
        "--opset",
        str(args.opset),
        "--sample-seq-len",
        str(args.sample_seq_len),
        "--sample-past-len",
        str(args.sample_past_len),
        *(["--disable-eager-attn"] if args.disable_eager_attn else []),
    )
    externalize_onnx_dir(output_dir)
    merge_shared_external_data(
        [
            output_dir / "moss_tts_prefill.onnx",
            output_dir / "moss_tts_decode_step.onnx",
        ],
        output_dir / "moss_tts_global_shared.data",
    )
    merge_shared_external_data(
        [
            output_dir / "moss_tts_local_cached_step.onnx",
            output_dir / "moss_tts_local_decoder.onnx",
            output_dir / "moss_tts_local_fixed_sampled_frame.onnx",
        ],
        output_dir / "moss_tts_local_shared.data",
    )
    patch_tts_external_data_files(output_dir / "tts_browser_onnx_meta.json")
    cleanup_redundant_external_data(output_dir)
    copy_tokenizer_model(checkpoint_path, output_dir)


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(Path(args.output_dir).expanduser().resolve())
    export_tts_onnx(args, output_dir)
    print(f"TTS-only ONNX export complete: {output_dir}")


if __name__ == "__main__":
    main()
