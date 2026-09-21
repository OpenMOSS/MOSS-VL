"""Expose the pinned NVIDIA wheels in the layout expected by JIT builders."""
import hashlib
import importlib.metadata
from pathlib import Path


def prepare(home: Path, sdk: Path | None = None) -> Path:
    if sdk is None:
        compiler = importlib.metadata.version("nvidia-cuda-nvcc")
        runtime = importlib.metadata.version("nvidia-cuda-runtime")
        if compiler.split(".")[:2] != runtime.split(".")[:2]:
            raise RuntimeError(f"CUDA compiler/runtime minor mismatch: {compiler} vs {runtime}")
        sdk = Path(importlib.metadata.distribution("nvidia-cuda-nvcc").locate_file("nvidia/cu13"))
    sdk = sdk.resolve()
    for required in ("bin/nvcc", "include/cuda_runtime.h", "lib/libcudart.so.13", "nvvm"):
        if not (sdk / required).exists():
            raise RuntimeError(f"Incomplete locked CUDA toolkit: {sdk / required}")
    # One checkout may be used by several virtual environments at once.
    sdk_id = hashlib.sha256(str(sdk).encode()).hexdigest()[:16]
    target = home.resolve() / "cuda-toolkit" / sdk_id
    target.mkdir(parents=True, exist_ok=True)

    def link(path: Path, source: Path):
        try:
            path.symlink_to(source)
        except FileExistsError:
            # Creation is atomic; concurrent launchers may share only this link.
            if not path.is_symlink() or path.resolve() != source.resolve():
                raise RuntimeError(
                    f"Unexpected CUDA toolkit entry; refusing to overwrite: {path}"
                ) from None

    for name in ("bin", "include", "nvvm", "cccl"):
        if (sdk / name).exists():
            link(target / name, sdk / name)
    lib = target / "lib64"
    lib.mkdir(exist_ok=True)
    for entry in (sdk / "lib").iterdir():
        link(lib / entry.name, entry)
    link(lib / "libcudart.so", sdk / "lib/libcudart.so.13")
    link(target / "lib", lib)
    return target


if __name__ == "__main__":
    print(prepare(Path(__file__).resolve().parents[2] / ".repro"))
