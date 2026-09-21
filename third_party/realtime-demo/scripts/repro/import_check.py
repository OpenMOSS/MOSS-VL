"""Check imports and CUDA without loading model weights."""
import importlib
import importlib.metadata
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

root = Path(__file__).resolve().parents[2]
if sys.argv[1] == "backend":
    from cuda_toolkit import prepare
    cuda_home = prepare(root / ".repro")
    os.environ.update(CUDA_HOME=str(cuda_home), CUDA_PATH=str(cuda_home))
sys.path.insert(0, str(root if sys.argv[1] == "demo" else root / ".repro/sglang-omni-main"))
packages = ["torch", "transformers", "torchcodec", "aiohttp", "websockets", "psutil"] if sys.argv[1] == "demo" else [
    "torch", "transformers", "sglang", "flashinfer", "sglang_omni", "sglang_omni.models.moss_vl_realtime.model_runner"]
for package in packages:
    importlib.import_module(package)
if sys.argv[1] == "backend":
    import flashinfer
    includes = Path(flashinfer.__file__).parent / "data/cccl/libcudacxx/include"
    with tempfile.TemporaryDirectory(dir=root / ".repro") as temporary:
        source = Path(temporary) / "compiler_check.cu"
        source.write_text("#include <cuda_runtime.h>\n#include <cooperative_groups.h>\n"
                          "__global__ void compiler_check(float* x) { x[0] += 1.0f; }\n")
        subprocess.run([str(cuda_home / "bin/nvcc"), "-std=c++17", "--cubin", f"-I{includes}",
                        str(source), "-o", str(Path(temporary) / "compiler_check.cubin")], check=True, timeout=120)
        linker_source = Path(temporary) / "linker_check.cpp"
        linker_source.write_text("#include <cuda_runtime.h>\n"
                                 "int main() { int version; return cudaRuntimeGetVersion(&version); }\n")
        executable = Path(temporary) / "linker_check"
        subprocess.run(["c++", str(linker_source), f"-I{cuda_home / 'include'}",
                        f"-L{cuda_home / 'lib64'}", f"-Wl,-rpath,{cuda_home / 'lib64'}",
                        "-lcudart", "-lcuda", "-o", str(executable)], check=True, timeout=120)
        subprocess.run([str(executable)], check=True, timeout=30)
import torch
if sys.argv[1] == "backend" and not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable in the newly installed environment")
print(json.dumps({"profile": sys.argv[1], "python": sys.version, "torch": torch.__version__,
                  "cuda": torch.version.cuda, "transformers": importlib.metadata.version("transformers"),
                  "cuda_home": os.environ.get("CUDA_HOME") if sys.argv[1] == "backend" else None}))
