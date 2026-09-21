"""Portable single-host deployment for the installed release profiles."""
import argparse
import csv
import errno
import fcntl
import json
import os
from pathlib import Path
import runpy
import socket
import subprocess
import sys
import time
import uuid
from urllib.request import ProxyHandler, build_opener

import psutil

ROOT = Path(__file__).resolve().parents[2]
HOME = ROOT / ".repro"
STATE = HOME / "run-state.json"
HTTP = build_opener(ProxyHandler({}))


def get(url):
    with HTTP.open(url, timeout=10) as response:
        data = response.read()
        return json.loads(data) if data else {}


def clean_environment():
    scanner = runpy.run_path(str(ROOT / "scripts/dev/check_env.py"))
    names = {name for _, name in scanner["scan_config"]() + scanner["scan_direct_readers"]()}
    names.update(scanner["SHELL_ONLY"])
    names.update(scanner["INTERNAL"])
    names.update({"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN",
                  "CUDA_VISIBLE_DEVICES", "AIGW_API_KEY", "AIGW_KEY_FILE", "NODE_OPTIONS", "NODE_PATH"})
    env = {k: v for k, v in os.environ.items() if k not in names and not k.startswith("SGLANG_")}
    (HOME / "home").mkdir(exist_ok=True)
    env.update(PATH=f"{HOME}/node/bin:{HOME}/tools/bin:{env['PATH']}", PYTHONNOUSERSITE="1",
               HOME=str(HOME / "home"), XDG_CACHE_HOME=str(HOME / "cache"),
               HF_HOME=str(HOME / "cache/huggingface"), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               ENV_DEPLOY_FILE="", MOSS_REPRO_ROOT=str(ROOT),
               FLASHINFER_WORKSPACE_BASE=str(HOME / "cache/flashinfer"),
               TORCHINDUCTOR_CACHE_DIR=str(HOME / "cache/torchinductor"),
               TORCH_EXTENSIONS_DIR=str(HOME / "cache/torch_extensions"), TORCH_HOME=str(HOME / "cache/torch"),
               TRITON_CACHE_DIR=str(HOME / "cache/triton"), CUDA_CACHE_PATH=str(HOME / "cache/cuda"))
    return env


def live_processes(processes):
    alive = []
    for process in processes:
        try:
            if process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
                alive.append(process)
        except psutil.NoSuchProcess:
            pass
    return alive


def wait_ports_released(ports, timeout=10):
    pending = set(ports)
    deadline = time.monotonic() + timeout
    if pending:
        print(f"Waiting for ports to be released: {sorted(pending)}", flush=True)
    while pending:
        for port in list(pending):
            try:
                with socket.socket() as listener:
                    # Match the server's address reuse behavior; TIME_WAIT is
                    # not a still-running listener.
                    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    listener.bind(("127.0.0.1", port))
            except OSError as exc:
                if exc.errno != errno.EADDRINUSE:
                    raise
            else:
                pending.remove(port)
        if not pending:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(f"Ports still occupied: {sorted(pending)}; state retained, startup not permitted")
        time.sleep(min(.1, max(0, deadline - time.monotonic())))


def wait_stopped(processes, timeout):
    deadline = time.monotonic() + timeout
    alive = live_processes(processes)
    while alive and time.monotonic() < deadline:
        _, alive = psutil.wait_procs(alive, timeout=min(.2, max(0, deadline - time.monotonic())))
        alive = live_processes(alive)
    return alive


def stop(records, ports=(), timeout=30):
    print("Stopping old services; waiting for their processes to exit...", flush=True)
    targets = {}
    for entry in records:
        token = entry.get("owner_token")
        try:
            p = psutil.Process(entry["pid"])
            if p.status() == psutil.STATUS_ZOMBIE:
                print(f"Process {p.pid} is already stopped (zombie); skipping environment check", flush=True)
            else:
                if abs(p.create_time() - entry["created"]) > .01 or p.environ().get("MOSS_REPRO_ROOT") != str(ROOT):
                    raise RuntimeError(f"Process ownership mismatch for pid {p.pid}; not stopped")
                targets[p.pid] = p
                targets.update({child.pid: child for child in p.children(recursive=True)})
        except psutil.NoSuchProcess:
            if not token:
                raise RuntimeError("Legacy deployment parent is gone; cannot verify orphan ownership. State retained for manual cleanup")
        if token:
            # Children retain this per-component token even after reparenting.
            for process in psutil.process_iter():
                try:
                    if process.status() == psutil.STATUS_ZOMBIE:
                        continue
                    env = process.environ()
                    if env.get("MOSS_REPRO_ROOT") == str(ROOT) and env.get("MOSS_REPRO_PROCESS_ID") == token:
                        targets[process.pid] = process
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
    targets = live_processes(targets.values())
    for process in targets:
        try: process.terminate()
        except psutil.NoSuchProcess: pass
    alive = wait_stopped(targets, timeout)
    if alive:
        print(f"Graceful stop timed out; killing owned processes: {[p.pid for p in alive]}", flush=True)
    for process in alive:
        try: process.kill()
        except psutil.NoSuchProcess: pass
    if wait_stopped(alive, 3):
        raise RuntimeError("Some owned processes have not exited; state retained")
    wait_ports_released(ports)
    print("Old processes stopped and deployment ports released.", flush=True)


def acquire_run_lock(lock, timeout=60):
    deadline = time.monotonic() + timeout
    announced = False
    while True:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except BlockingIOError:
            if not announced:
                print("Waiting for the previous startup/shutdown operation to finish...", flush=True)
                announced = True
            if time.monotonic() >= deadline:
                raise RuntimeError("Previous startup/shutdown is still running; no new operation started")
            time.sleep(min(.1, max(0, deadline - time.monotonic())))


def wait_for(url, process, ready, timeout=1200):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"{url}: child exited with {process.returncode}; inspect .repro/logs")
        try:
            if ready(get(url)): return
        except (OSError, ValueError): pass
        time.sleep(2)
    raise TimeoutError(f"Service did not become ready: {url}")


def up(args):
    install = json.loads((HOME / "managed-install.json").read_text())
    if install["state"] != "installed" or not (HOME / "models-verified.json").exists():
        raise RuntimeError("Run bootstrap including model downloads first")
    if STATE.exists():
        raise RuntimeError("Deployment state already exists; use status/down before starting")
    profiles = set(install["profiles"])
    if "memory" in profiles and args.main_gpu == args.memory_gpu:
        raise RuntimeError("This validated profile requires separate VLM and memory GPUs")
    rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.total,memory.free",
                                    "--format=csv,noheader,nounits"], text=True)
    devices = {int(row[0]): (int(row[1]), int(row[2])) for row in csv.reader(rows.splitlines())}
    requested = [(args.main_gpu, args.main_memory_fraction)]
    if "memory" in profiles: requested.append((args.memory_gpu, args.memory_fraction))
    for gpu, fraction in requested:
        if gpu not in devices or not 0 < fraction < 1:
            raise ValueError("Invalid GPU index or memory fraction")
        total, free = devices[gpu]
        if not args.allow_shared_gpus and total - free > 2048:
            raise RuntimeError(f"GPU {gpu} is already in use; choose an idle GPU or explicitly allow sharing")
        if free < total * fraction + 4096:
            raise RuntimeError(f"GPU {gpu} lacks free memory for the requested fraction and startup reserve")
    ports = {name: args.base_port + offset for offset, name in enumerate(["backend", "api", "web", "pi", "memory", "tts"])}
    if not 1024 <= args.base_port <= 65530:
        raise ValueError("base-port must be between 1024 and 65530")
    for port in ports.values():
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", port))  # never steal someone else's port
    env = clean_environment()
    cuda_home = subprocess.check_output([str(HOME / ".venv-main/bin/python"),
        str(ROOT / "scripts/repro/cuda_toolkit.py")], text=True).strip()
    if not (Path(cuda_home) / "bin/nvcc").is_file():
        raise RuntimeError("The locked CUDA compiler wheel is missing; rerun bootstrap")
    env.update(CUDA_HOME=cuda_home, CUDA_PATH=cuda_home, PATH=f"{cuda_home}/bin:{env['PATH']}")
    model = HOME / "models/vlm"
    env.update(VLM_DEPLOY="sglang_omni", MODEL_PATH=str(model), AUTOLOAD_VLM="1", OFFLINE_PROVIDER="none",
        SGLANG_OMNI_URLS=f"http://127.0.0.1:{ports['backend']}", SGLANG_OMNI_SESSIONS_PER_REPLICA="4",
        SGLANG_OMNI_CONTEXT_LENGTH="131072", GEN_MAX_TOKENS_PER_TURN="86400", WS_MAX_SIZE="67108864",
        ASR_ENABLED=str(int("asr" in profiles)), TTS_ENABLED=str(int("tts" in profiles)), TTS_SPAWN="0",
        MEMORY_ENABLED=str(int("memory" in profiles)), DATA_DIR=str(HOME / "data"),
        MOSS_LOG_FILE=str(HOME / "logs/backend-handler.log"), VLM_WORKER_LOG_DIR=str(HOME / "logs/workers"),
        MOSS_TTS_NANO_OUTPUT_DIR=str(HOME / "data/tts"),
        SENSEVOICE_MODEL=str(HOME / "models/asr/SenseVoiceSmall"), SENSEVOICE_VAD_MODEL=str(HOME / "models/asr/fsmn-vad"),
        ASR_DEVICE="cpu", ASR_FP16="0", MEMORY_EMBED_DEVICE="cpu",
        MEMORY_EMBED_TEXT_MODEL=str(HOME / "models/memory/bge-m3"),
        MEMORY_EMBED_IMAGE_MODEL=str(HOME / "models/memory/chinese-clip"),
        MEMORY_PI_URL=f"http://127.0.0.1:{ports['pi']}", MEMORY_SUMMARY_PROVIDER="pi", MEMORY_DECISION_MODE="hybrid",
        MEMORY_PI_DECIDE_TIMEOUT_S="8", MEMORY_PI_COMPACT_TIMEOUT_S="120", TTS_PROVIDER="moss_tts_nano",
        MOSS_TTS_NANO_BASE_URL=f"http://127.0.0.1:{ports['tts']}", TTS_SIDECAR_COUNT="1",
        MOSS_TTS_NANO_ONNX_MODEL_DIR=str(HOME / "models/tts"), MOSS_TTS_NANO_BACKEND="onnx",
        MOSS_TTS_NANO_ONNX_CPU_THREADS=str(args.cpu_threads), OMP_NUM_THREADS=str(args.cpu_threads),
        MKL_NUM_THREADS=str(args.cpu_threads), MAX_JOBS=str(args.cpu_threads),
        MOSS_TTS_NANO_DEVICE="cpu", MOSS_TTS_NANO_VOICE="Junhao", TTS_VOICE="Junhao",
        VITE_BACKEND_ORIGIN=f"http://127.0.0.1:{ports['api']}", AIGW_AUTH_MODE="local", PI_AGENT_MODE="hop",
        AIGW_DECIDE_BASE_URL=f"http://127.0.0.1:{ports['memory']}/v1", AIGW_COMPACT_BASE_URL=f"http://127.0.0.1:{ports['memory']}/v1",
        AIGW_DECIDE_MODEL="Qwen3-4B-Instruct-2507", AIGW_COMPACT_MODEL="Qwen3-4B-Instruct-2507",
        AIGW_LOCAL_NO_REASONING="1", PI_PORT=str(ports["pi"]), PI_CONTEXT_TOKENS="16384",
        PI_COMPACT_CHUNK_TOKENS="4096", PI_COMPACT_TIMEOUT_MS="60000", PI_DECIDE_TIMEOUT_MS="7000")
    records = []
    state = {"profiles": sorted(profiles), "ports": ports, "processes": records}
    (HOME / "logs").mkdir(exist_ok=True)

    def start(name, command, cwd=ROOT, overrides=None):
        child_env = dict(env, **(overrides or {}))
        owner_token = uuid.uuid4().hex
        child_env["MOSS_REPRO_PROCESS_ID"] = owner_token
        child_env["PATH"] = f"{Path(command[0]).parent}:{child_env['PATH']}"
        with (HOME / f"logs/{name}.log").open("a") as log:
            process = subprocess.Popen(list(map(str, command)), cwd=cwd, env=child_env,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        records.append({"name": name, "pid": process.pid, "created": psutil.Process(process.pid).create_time(),
                        "owner_token": owner_token})
        STATE.write_text(json.dumps(state, indent=2))
        print(f"Starting {name} (pid {process.pid})", flush=True)
        return process

    backend = HOME / "sglang-omni-main"
    python = HOME / ".venv-main/bin/python"
    try:
        main = start("backend", [python, "examples/run_moss_vl_realtime_server.py", "--model-path", model,
            "--gpu", args.main_gpu, "--host", "127.0.0.1", "--port", ports["backend"],
            "--context-length", "131072", "--mem-fraction-static", args.main_memory_fraction,
            "--max-running-requests", "4"], cwd=backend)
        wait_for(f"http://127.0.0.1:{ports['backend']}/health", main, lambda d: d.get("status") == "healthy")
        if "memory" in profiles:
            memory = start("memory", [python, "-m", "sglang.launch_server", "--model-path", HOME / "models/memory/qwen3-4b",
                "--served-model-name", "Qwen3-4B-Instruct-2507",
                "--host", "127.0.0.1", "--port", ports["memory"], "--context-length", "16384",
                "--mem-fraction-static", args.memory_fraction, "--max-running-requests", "4",
                "--max-total-tokens", "65536", "--disable-radix-cache"], cwd=backend,
                overrides={"CUDA_VISIBLE_DEVICES": str(args.memory_gpu)})
            wait_for(f"http://127.0.0.1:{ports['memory']}/health", memory, lambda d: True)
            pi = start("pi", [HOME / "node/bin/node", "service.mjs"], cwd=ROOT / "services/pi_agent")
            wait_for(f"http://127.0.0.1:{ports['pi']}/ready", pi, lambda d: d.get("ok") is True, timeout=60)
        if "tts" in profiles:
            tts = start("tts", [ROOT / ".venv/bin/python", "-m", "uvicorn", "moss_tts_nano_sidecar:app",
                "--host", "127.0.0.1", "--port", ports["tts"]],
                cwd=ROOT / "server/adapters/tts/moss_tts_nano/sidecar/backend")
            wait_for(f"http://127.0.0.1:{ports['tts']}/health", tts, lambda d: d.get("ready") is True or d.get("status") == "ok")
        api = start("api", [ROOT / ".venv/bin/python", "-m", "uvicorn", "server.app:app", "--host", "127.0.0.1",
            "--port", ports["api"], "--ws-max-size", "67108864"])
        wait_for(f"http://127.0.0.1:{ports['api']}/api/status", api, lambda d: d.get("vlm", {}).get("loaded") is True)
        web = start("web", [HOME / "node/bin/node", ROOT / "node_modules/vite/bin/vite.js", "preview",
            "--host", "127.0.0.1", "--port", ports["web"], "--strictPort"])
        for _ in range(30):
            if web.poll() is not None: raise RuntimeError("web exited")
            try:
                with HTTP.open(f"http://127.0.0.1:{ports['web']}/", timeout=3): break
            except OSError: time.sleep(1)
        else: raise TimeoutError("web did not start")
        print(f"Ready: http://localhost:{ports['web']} (remote cameras require HTTPS)", flush=True)
        print(f"API: http://127.0.0.1:{ports['api']} | native backend: http://127.0.0.1:{ports['backend']}", flush=True)
        print(f"curl --fail http://127.0.0.1:{ports['api']}/api/status", flush=True)
        print(f"curl --fail http://127.0.0.1:{ports['api']}/v1/realtime/health", flush=True)
        if 'memory' in profiles:
            print(f"Memory pi: http://127.0.0.1:{ports['pi']}", flush=True)
        if 'tts' in profiles:
            print(f"TTS: http://127.0.0.1:{ports['tts']}", flush=True)
    except BaseException:
        stop(records, ports.values())
        STATE.unlink(missing_ok=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["up", "down", "status"])
    parser.add_argument("--main-gpu", type=int, default=0)
    parser.add_argument("--memory-gpu", type=int, default=1)
    parser.add_argument("--base-port", type=int, default=18500)
    parser.add_argument("--main-memory-fraction", type=float, default=.5)
    parser.add_argument("--memory-fraction", type=float, default=.2)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--allow-shared-gpus", action="store_true")
    args = parser.parse_args()
    if args.cpu_threads < 1:
        raise ValueError("cpu-threads must be positive")
    HOME.mkdir(exist_ok=True)
    lock = (HOME / "run.lock").open("a")
    acquire_run_lock(lock)
    if args.command == "up": up(args)
    elif args.command == "down":
        if STATE.exists():
            state = json.loads(STATE.read_text())
            stop(state["processes"], state.get("ports", {}).values())
            STATE.unlink()
            print("Stopped; deployment state cleared. A new startup may now begin.", flush=True)
        else:
            print("Not running; no deployment state to clear.", flush=True)
    elif STATE.exists():
        state = json.loads(STATE.read_text())
        print(json.dumps({"state": state, "gateway": get(f"http://127.0.0.1:{state['ports']['api']}/api/status")}, ensure_ascii=False))
    else: print("Not running")


if __name__ == "__main__": main()
