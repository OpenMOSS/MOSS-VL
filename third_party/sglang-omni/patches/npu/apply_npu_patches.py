"""Apply known Ascend environment patches, rejecting incompatible sources."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import shutil
import subprocess
import tempfile
from pathlib import Path


def detect_patch_set(site: Path) -> str:
    path = site / "sglang/srt/model_executor/model_runner.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "ModelRunner":
            for method in node.body:
                if isinstance(method, ast.FunctionDef) and method.name == "__init__":
                    names = {
                        arg.arg for arg in method.args.args + method.args.kwonlyargs
                    }
                    if "ps" in names:
                        return "0.5.16"
                    if {"tp_rank", "tp_size"} <= names:
                        return "0.5.14"
    raise RuntimeError("cannot identify the installed SGLang ModelRunner API")


def run_patch(staging: Path, patch: bytes, target: str | None, *options):
    # -N also applies to reverse probes: never let GNU patch silently ignore -R.
    command = ["patch", "--batch", "--silent", "--forward", "-p1"]
    if target:
        command.append(target)
    return subprocess.run(
        command + list(options), input=patch, cwd=staging, capture_output=True
    )


def apply_patch_set(staging: Path, here: Path, operations) -> None:
    if any(name == "sglang-0.5.14.patch" for name, _ in operations):
        # The older combined patch contains a guard removed by the supplement.
        # Normalize this known state privately before its idempotence check.
        supplement = (here / "0008-0.5.14-attention-alignment.patch").read_bytes()
        if (
            run_patch(staging, supplement, None, "--reverse", "--dry-run").returncode
            == 0
        ):
            if run_patch(staging, supplement, None, "--reverse").returncode:
                raise RuntimeError(
                    "failed normalizing 0.5.14 patches; no installed files were changed"
                )
    for name, target in operations:
        patch = (here / name).read_bytes()
        if run_patch(staging, patch, target, "--forward", "--dry-run").returncode == 0:
            applied = run_patch(staging, patch, target, "--forward")
            if applied.returncode:
                raise RuntimeError(f"failed applying {name}: {applied.stderr.decode()}")
            print(f"Applied: {name}")
        elif (
            run_patch(staging, patch, target, "--reverse", "--dry-run").returncode == 0
        ):
            print(f"Already applied: {name}")
        elif name == "0007-ascend-npu-attention-integrated-fixes.patch":
            migration = (here / "upgrade-0.5.16-attention.patch").read_bytes()
            if run_patch(staging, migration, None, "--forward", "--dry-run").returncode:
                raise RuntimeError(
                    f"{name} is incompatible with the installed sources. "
                    "No installed files were changed."
                )
            result = run_patch(staging, migration, None, "--forward")
            if (
                result.returncode
                or run_patch(
                    staging, patch, target, "--reverse", "--dry-run"
                ).returncode
            ):
                raise RuntimeError(
                    "attention upgrade failed validation; no installed files were changed"
                )
            print("Upgraded: legacy 0.5.16 attention patches")
        else:
            raise RuntimeError(
                f"{name} is incompatible with the installed sources. "
                "No installed files were changed; use the documented SGLang build."
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("site_packages", nargs="?", type=Path)
    parser.add_argument(
        "--patch-set",
        choices=("auto", "0.5.14", "0.5.16"),
        default="auto",
        help="select source layout explicitly for vendor backports",
    )
    args = parser.parse_args()
    if args.site_packages is None:
        spec = importlib.util.find_spec("sglang")
        if spec is None or not spec.origin:
            parser.error("sglang is not installed in this interpreter")
        site = Path(spec.origin).resolve().parent.parent
    else:
        site = args.site_packages.resolve()
    if not (site / "sglang").is_dir():
        parser.error(f"expected a site-packages directory containing sglang: {site}")
    if shutil.which("patch") is None:
        parser.error("GNU patch is required")

    here = Path(__file__).resolve().parent
    native = "sglang/srt/hardware_backend/npu/attention/ascend_torch_native_backend.py"
    backend = "sglang/srt/hardware_backend/npu/attention/ascend_backend.py"
    moss = "sglang/srt/models/moss_vl.py"
    vision = "sglang/srt/layers/attention/vision.py"
    # 0007 replaces 0002/0003; 0004 still supplies the startup capability marker.
    operations = [
        ("0001-fix-vision-rope-for-transformers-5-and-npu-inv-freq.patch", moss),
        ("0007-ascend-npu-attention-integrated-fixes.patch", None),
        ("0005-chunk-vision-encoder-for-multi-frame-rounds.patch", moss),
        ("0006-ascend-vision-flash-attention-backend.patch", None),
    ]
    patch_set = detect_patch_set(site) if args.patch_set == "auto" else args.patch_set
    if patch_set == "0.5.14":
        operations = [
            ("sglang-0.5.14.patch", None),
            ("0008-0.5.14-attention-alignment.patch", None),
            ("0005-chunk-vision-encoder-for-multi-frame-rounds.patch", moss),
            ("0006-0.5.14-vision-backend.patch", moss),
        ]
    operations.append(("0004-use-torch-cross-attention.patch", None))
    print(f"SGLang patch set: {patch_set}")
    targets = (
        (moss, native, backend, vision)
        if patch_set == "0.5.16"
        else (moss, native, backend)
    )
    originals = {path: (site / path).read_bytes() for path in targets}
    # Resolve all patches on private copies before writing any installed file.
    with tempfile.TemporaryDirectory(prefix="moss-npu-patches-") as temp:
        staging = Path(temp)
        for path, data in originals.items():
            destination = staging / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
        apply_patch_set(staging, here, operations)
        updated = {path: (staging / path).read_bytes() for path in originals}
        for path, data in updated.items():
            ast.parse(data, filename=path)
        for path, before in originals.items():
            if (site / path).read_bytes() != before:
                raise RuntimeError(f"installed source changed concurrently: {path}")
        for path, data in updated.items():
            if data != originals[path]:
                destination = site / path
                backup = destination.with_suffix(destination.suffix + ".moss-npu.bak")
                if not backup.exists():
                    backup.write_bytes(originals[path])
                destination.write_bytes(data)
    print(f"Ascend patches installed under {site}; restart serving processes.")


if __name__ == "__main__":
    main()
