# Backend Dependencies and Containers

**English** | [简体中文](./README_zh.md)

`requirements.lock` pins backend dependencies for Linux / Python 3.12. Follow the [installation guide](../../docs/get_started/installation.md) for native setup. For the complete application, start with the [Demo](https://github.com/fnlp-vision/MOSS-VL-Realtime_Demo#readme).

| File | Purpose |
| --- | --- |
| [requirements.lock](./requirements.lock) | Package versions and release hashes |
| [build-constraints.txt](./build-constraints.txt) | Build dependency constraints |
| [cuda-toolchain.in](./cuda-toolchain.in) | CUDA compiler and runtime requirements |
| [cuda_toolkit.py](./cuda_toolkit.py) | Environment-specific CUDA view for JIT builders |

## Containers

The Dockerfile uses source from the build context; startup does not clone or update code. Mount model weights at runtime and keep credentials out of the image.

```bash
docker build -f docker/Dockerfile -t moss-vl-realtime:local .
docker run --rm --gpus all --ipc=host -p 18500:18500 \
  -v /absolute/model:/models/moss-vl:ro moss-vl-realtime:local \
  python3 examples/run_moss_vl_realtime_server.py --model-path /models/moss-vl \
  --gpu 0 --host 0.0.0.0 --port 18500 --mem-fraction-static 0.5 --max-running-requests 4
```

Requires Docker and NVIDIA Container Toolkit. The container path is not validated; verify the build and GPU execution before use. Do not expose internal services directly to the public Internet.
