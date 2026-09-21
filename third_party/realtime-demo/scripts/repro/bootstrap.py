"""Install current repository sources with isolated, locked runtime dependencies."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "deployment/repro/manifest.json"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command(args, *, env=None, cwd=ROOT):
    print("+", " ".join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), cwd=cwd, env=env, check=True)


def environment(home):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PIP_", "UV_", "NPM_CONFIG_", "npm_config_"))
           and k not in {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "NODE_OPTIONS", "NODE_PATH"}}
    env.update(UV_NO_CONFIG="1", UV_CACHE_DIR=str(home / "cache/uv"),
               UV_PYTHON_INSTALL_DIR=str(home / "python"), UV_PYTHON_BIN_DIR=str(home / "bin"), PYTHONNOUSERSITE="1",
               UV_HTTP_TIMEOUT="600", UV_HTTP_RETRIES="5", UV_CONCURRENT_DOWNLOADS="4",
               PIP_CONFIG_FILE=os.devnull, PIP_INDEX_URL="https://pypi.org/simple",
               HF_HOME=str(home / "cache/huggingface"), HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
               HF_HUB_DOWNLOAD_TIMEOUT="600", HF_HUB_ETAG_TIMEOUT="30")
    env.update(NPM_CONFIG_CACHE=str(home / "cache/npm"), NPM_CONFIG_USERCONFIG=os.devnull,
               NPM_CONFIG_GLOBALCONFIG=str(home / "npm-empty-global.rc"))
    return env


def doctor(required_gib=60):
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("This release profile supports Linux x86_64 only")
    missing = [name for name in ("git", "gcc", "g++", "cmake", "ffmpeg", "nvidia-smi") if not shutil.which(name)]
    if missing:
        raise RuntimeError("Missing prerequisites: " + ", ".join(missing) + "; see deployment/repro/README.md")
    command(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])
    drivers = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).splitlines()
    if not drivers or any(int(version.split(".")[0]) < 580 for version in drivers):
        raise RuntimeError("CUDA 13 requires an R580-or-newer driver; JIT compatibility is checked during GPU smoke")
    if shutil.disk_usage(ROOT).free < required_gib * 1024**3:
        raise RuntimeError(f"At least {required_gib} GiB free space is required for this profile")


def install_tools(home, spec, env):
    tools = home / "tools"
    if not (tools / "bin/python").exists():
        command([sys.executable, "-m", "venv", tools], env=env)
    command([tools / "bin/python", "-m", "pip", "--isolated", "install", "--require-hashes",
             "--cache-dir", home / "cache/pip", "--index-url", "https://pypi.org/simple",
             "-r", ROOT / "deployment/repro/locks/tools.lock"], env=env)
    uv = tools / "bin/uv"
    command([uv, "python", "install", spec["python"]], env=env)
    node = home / "node"
    if not (node / "bin/node").exists():
        archive = home / "node.tar.xz"
        with urlopen(spec["node"]["url"], timeout=120) as source, archive.open("wb") as target:
            shutil.copyfileobj(source, target)
        if sha256(archive) != spec["node"]["sha256"]:
            raise RuntimeError("Node archive checksum mismatch")
        with tempfile.TemporaryDirectory(dir=home) as temporary:
            with tarfile.open(archive) as bundle:
                bundle.extractall(temporary, filter="data")
            shutil.move(str(next(Path(temporary).iterdir())), node)
        archive.unlink()
    actual = subprocess.check_output([node / "bin/node", "--version"], text=True).strip()
    if actual != "v" + spec["node"]["version"]:
        raise RuntimeError("Installed Node does not match the release manifest")
    env["PATH"] = f"{node}/bin:{tools}/bin:{env['PATH']}"
    return uv


def backend_source(home, spec, provided, env, *, update=False):
    target = home / "sglang-omni-main"
    if provided:
        source = Path(provided).resolve()
        if not (source / "pyproject.toml").is_file():
            raise RuntimeError("--backend-source must contain pyproject.toml")
        inventory = source / "source-manifest.json"
        if not inventory.exists():
            raise RuntimeError("--backend-source requires the release's source-manifest.json")
        provenance = json.loads(inventory.read_text())
        for name, digest in provenance["files"].items():
            file = source / name
            if Path(name).is_absolute() or ".." in Path(name).parts or not file.is_file() or sha256(file) != digest:
                raise RuntimeError(f"Backend bundle checksum mismatch: {name}")
        # Copy only the verified inventory; do not copy environments or logs.
        if not target.exists():
            with tempfile.TemporaryDirectory(dir=home) as temporary:
                staged = Path(temporary) / "backend"
                staged.mkdir()
                for name in provenance["files"]:
                    destination = staged / name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source / name, destination)
                shutil.copy2(inventory, staged / inventory.name)
                staged.rename(target)
        else:
            for name, digest in provenance["files"].items():
                if not (target / name).is_file() or sha256(target / name) != digest:
                    raise RuntimeError(f"Existing backend bundle modified: {name}")
    elif not target.exists():
        command(["git", "init", target], env=env)
        command(["git", "-C", target, "remote", "add", "origin", spec["backend"]["url"]], env=env)
    if not provided and (target / ".git").exists():
        head = subprocess.run(["git", "-C", str(target), "rev-parse", "--verify", "HEAD"],
                              capture_output=True, text=True, check=False)
        if head.returncode == 0:
            if not update:
                return target
            changed = subprocess.check_output(["git", "-C", target, "status", "--porcelain"], text=True)
            if changed.strip():
                raise RuntimeError("Backend has local changes; refusing to overwrite them")
            marker = home / "managed-install.json"
            previous = json.loads(marker.read_text()).get("backend_revision") if marker.exists() else None
            if previous and previous != head.stdout.strip():
                raise RuntimeError("Backend HEAD was changed locally; use a separate checkout")
        # An interrupted first fetch leaves an initialized repo without HEAD.
        command(["git", "-C", target, "fetch", "--depth", "1", spec["backend"]["url"], "HEAD"], env=env)
        command(["git", "-C", target, "checkout", "--detach", "FETCH_HEAD"], env=env)
    elif not provided:
        raise RuntimeError("Existing backend lacks provenance; specify its verified --backend-source")
    return target


def backend_revision(backend):
    if (backend / ".git").exists():
        return subprocess.check_output(["git", "-C", backend, "rev-parse", "HEAD"], text=True).strip()
    return json.loads((backend / "source-manifest.json").read_text())["revision"]


def install_python(uv, home, backend, env):
    locks = ROOT / "deployment/repro/locks"
    for target, lock, constraints in (
        (ROOT / ".venv", locks / "demo.lock", locks / "build-constraints.txt"),
        (home / ".venv-main", backend / "deployment/repro/requirements.lock",
         backend / "deployment/repro/build-constraints.txt"),
    ):
        if not (target / "bin/python").exists():
            command([uv, "venv", "--python", "3.12.12", target], env=env)
        command([uv, "pip", "sync", "--python", target / "bin/python", "--require-hashes",
                 "--build-constraint", constraints,
                 "--index-url", "https://pypi.org/simple", *(["--torch-backend", "cpu"] if lock.name == "demo.lock" else []),
                 lock], env=env)
        command([uv, "pip", "check", "--python", target / "bin/python"], env=env)
    command([uv, "pip", "install", "--python", home / ".venv-main/bin/python", "--no-deps",
             "--no-build-isolation", "-e", backend], env=env)
    command([uv, "pip", "check", "--python", home / ".venv-main/bin/python"], env=env)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-memory", action="store_true")
    parser.add_argument("--with-asr", action="store_true")
    parser.add_argument("--with-tts", action="store_true", help="pinned MOSS-TTS-Nano ONNX CPU profile")
    parser.add_argument("--backend-source", type=Path)
    parser.add_argument("--update", action="store_true", help="refresh backend and selected models from their repositories")
    parser.add_argument("--doctor-only", action="store_true")
    parser.add_argument("--check-gpu", type=int, default=0, help="GPU used for the import/driver check, not deployment placement")
    parser.add_argument("--skip-model-download", action="store_true", help="explicitly incomplete/offline preparation")
    args = parser.parse_args()
    doctor(120 if args.with_memory or args.with_asr or args.with_tts else 60)
    if args.doctor_only:
        return
    home = ROOT / ".repro"
    home.mkdir(exist_ok=True)
    run_lock = (home / "run.lock").open("a")
    fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (home / "run-state.json").exists():
        raise RuntimeError("Stop the managed deployment with run.py down before installing")
    lock = (home / "install.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    marker = home / "managed-install.json"
    if (ROOT / ".venv").exists() and not marker.exists():
        raise RuntimeError("Refusing to modify an existing unmanaged .venv; use a fresh checkout")
    spec = json.loads(SPEC.read_text())
    env = environment(home)
    env["CUDA_VISIBLE_DEVICES"] = str(args.check_gpu)
    previous = json.loads(marker.read_text()) if marker.exists() else {}
    profiles = sorted(set(previous.get("profiles", [])) | {"base"} |
                      {name for name in ("memory", "asr", "tts") if getattr(args, "with_" + name)})
    marker.write_text(json.dumps({**previous, "state": "installing", "profiles": profiles,
                                  "manifest_sha256": sha256(SPEC)}, indent=2))
    uv = install_tools(home, spec, env)
    backend = backend_source(home, spec, args.backend_source, env, update=args.update)
    marker.write_text(json.dumps({**previous, "state": "installing", "profiles": profiles,
                                  "backend_revision": backend_revision(backend),
                                  "manifest_sha256": sha256(SPEC)}, indent=2))
    install_python(uv, home, backend, env)
    command([ROOT / ".venv/bin/python", ROOT / "scripts/repro/import_check.py", "demo"], env=env)
    command([home / ".venv-main/bin/python", ROOT / "scripts/repro/import_check.py", "backend"], env=env)
    for directory in (ROOT, ROOT / "services/pi_agent"):
        command([home / "node/bin/npm", "ci", "--registry", "https://registry.npmjs.org"], cwd=directory, env=env)
    command([home / "node/bin/npm", "run", "build"], env=env)
    if not args.skip_model_download:
        command([home / ".venv-main/bin/python", ROOT / "scripts/repro/models.py", "--profiles", *profiles,
                 *(["--update"] if args.update else [])], env=env)
    state = {"state": "installed" if not args.skip_model_download else "dependencies-only", "profiles": profiles,
             "manifest_sha256": sha256(SPEC), "backend_revision": backend_revision(backend),
             "backend_url": spec["backend"]["url"],
             "backend_lock_sha256": sha256(backend / "deployment/repro/requirements.lock"),
             "locks": {p.name: sha256(p) for p in (ROOT / "deployment/repro/locks").iterdir() if p.is_file()},
             "npm_locks": {name: sha256(ROOT / name) for name in ["package-lock.json", "services/pi_agent/package-lock.json"]}}
    marker.write_text(json.dumps(state, indent=2))
    print(json.dumps(state), flush=True)
    print("Next: .venv/bin/python scripts/repro/run.py up --main-gpu 0 --memory-gpu 1", flush=True)


if __name__ == "__main__":
    main()
