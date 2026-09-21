"""Reconcile memory services; never claim or kill an unowned listener."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
from urllib.request import ProxyHandler, build_opener

import psutil
from stop_backend import is_owned, require_free_port, stop_processes

HTTP = build_opener(ProxyHandler({}))


def get(base, path):
    with HTTP.open(base + path, timeout=3) as response:
        data = response.read()
        return json.loads(data) if data else {}


def implementation(directory):
    digest = hashlib.sha256()
    for file in sorted(directory.glob("*.mjs")):
        if file.name.endswith(".test.mjs"):
            continue
        digest.update(file.name.encode())
        digest.update(file.read_bytes())
    return digest.hexdigest()


def config(env):
    root = Path(env["OMNI_ROOT"])
    port = int(env.get("DECIDE_LLM_PORT", "38090"))
    desired = {
        "model_path": env.get("DECIDE_LLM_MODEL") or "/inspire/hdd/project/video-understanding/public/share/models/Qwen3-4B-Instruct-2507",
        "mem_fraction_static": float(env.get("DECIDE_LLM_MEM_FRAC", "0.2")),
        "context_length": int(env.get("DECIDE_LLM_CONTEXT_LENGTH", "16384")),
        "max_running_requests": int(env.get("DECIDE_LLM_MAX_RUNNING_REQUESTS", "4")),
        "max_total_tokens": int(env.get("DECIDE_LLM_MAX_TOTAL_TOKENS", "65536")),
    }
    if not 0 < desired["mem_fraction_static"] < 1 or any(
            desired[k] <= 0 for k in ("context_length", "max_running_requests", "max_total_tokens")):
        raise ValueError("invalid memory backend resource configuration")
    gpu = env.get("DECIDE_LLM_GPU", "1")
    node = env.get("PI_AGENT_NODE") or "/inspire/hdd/project/video-understanding/public/personal/yxchen/.local/node-v22.12.0-linux-x64/bin/node"
    return root, port, desired, gpu, node


def backend_differences(base, desired, gpu):
    get(base, "/health")  # HTTP errors fail, empty HTTP 200 is valid.
    info = get(base, "/get_server_info")
    differences = {key: {"actual": info.get(key), "wanted": value}
                   for key, value in desired.items() if info.get(key) != value}
    port = int(base.rsplit(":", 1)[1])
    listeners = {conn.pid for conn in psutil.net_connections(kind="tcp")
                 if conn.status == psutil.CONN_LISTEN and conn.laddr.port == port and conn.pid}
    masks = []
    for pid in listeners:
        try:
            masks.append(psutil.Process(pid).environ().get("CUDA_VISIBLE_DEVICES"))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    if not masks or any(mask != gpu for mask in masks):
        differences["CUDA_VISIBLE_DEVICES"] = {"actual": masks, "wanted": gpu}
    return differences


def stop_owned(repo, role, port):
    processes = [p for p in psutil.process_iter() if is_owned(p, repo, [role], port)]
    # Capture children before stopping the launcher: SGLang changes process
    # titles, which can make the original environment unavailable in /proc.
    children = {}
    for process in processes:
        try:
            children.update({p.pid: p for p in process.children(recursive=True)})
        except psutil.NoSuchProcess:
            pass
    processes = list({p.pid: p for p in [*processes, *children.values()]}.values())
    survivors = stop_processes(processes)
    if survivors:
        raise RuntimeError(f"owned {role} processes have not exited")
    require_free_port(port)


def spawn(command, cwd, env, log):
    with log.open("a") as output:
        return subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                stdout=output, stderr=subprocess.STDOUT, start_new_session=True)


def wait_ready(check, process, seconds):
    deadline = time.monotonic() + seconds
    last = "not ready"
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"child exited during startup: {process.returncode}")
        try:
            if check():
                return
        except (OSError, ValueError) as exc:
            last = str(exc)
        time.sleep(min(1, max(0, deadline - time.monotonic())))
    raise TimeoutError(f"readiness deadline exceeded: {last}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    env = dict(os.environ)
    root, port, desired, gpu, node = config(env)
    base = f"http://127.0.0.1:{port}"
    pi_port = int(env.get("PI_PORT") or 38082)
    pi_base = f"http://127.0.0.1:{pi_port}"
    logs = Path(env.get("LOG_DIR") or repo / "logs/pi_agent")
    logs.mkdir(parents=True, exist_ok=True)
    wait = float(env.get("MEMORY_BACKEND_WAIT_S") or 1200)
    if not math.isfinite(wait) or wait <= 0:
        raise ValueError("MEMORY_BACKEND_WAIT_S must be positive and finite")
    try:
        differences = backend_differences(base, desired, gpu)
    except (OSError, ValueError):
        differences = {"ready": False}
    if differences or env.get("FORCE_4B") == "1":
        print(f"4B configuration requires update: {json.dumps(differences)}", flush=True)
        if env.get("START_4B", "0") != "1":
            raise RuntimeError("4B unavailable/mismatched; START_4B=1 is required to reconcile owned processes")
        stop_owned(repo, "memory-llm", port)
        child_env = dict(env, MOSS_DEPLOY_REPO=str(repo), MOSS_DEPLOY_ROLE="memory-llm",
                         MOSS_DEPLOY_PORT=str(port), CUDA_VISIBLE_DEVICES=gpu,
                         PATH=f"{root}/.venv-main/bin:{env.get('PATH', '')}")
        command = [str(root / ".venv-main/bin/python"), "-m", "sglang.launch_server",
                   "--host", "127.0.0.1", "--port", str(port), "--disable-radix-cache"]
        for key, value in desired.items():
            flag = "--model-path" if key == "model_path" else "--" + key.replace("_", "-")
            command.extend([flag, str(value)])
        process = spawn(command, root / "sglang-omni-main", child_env, logs / "decide_llm.log")
        try:
            wait_ready(lambda: not backend_differences(base, desired, gpu), process, wait)
        except Exception:
            stop_owned(repo, "memory-llm", port)
            raise
    else:
        print("4B ready; requested model, GPU and resource settings match", flush=True)

    defaults = {
        "PI_PORT": str(pi_port), "AIGW_DECIDE_BASE_URL": base + "/v1",
        "AIGW_COMPACT_BASE_URL": base + "/v1", "AIGW_DECIDE_MODEL": Path(desired["model_path"]).name,
        "AIGW_COMPACT_MODEL": Path(desired["model_path"]).name, "AIGW_LOCAL_NO_REASONING": "1",
        "PI_CONTEXT_TOKENS": str(desired["context_length"]), "PI_COMPACT_CHUNK_TOKENS": "4096",
        "PI_COMPACT_TIMEOUT_MS": "60000", "PI_DECIDE_TIMEOUT_MS": "7000", "PI_AGENT_MODE": "hop",
        "PI_AGENT_TIMEOUT_MS": "45000", "BOARD_MEMORY_URL": "http://127.0.0.1:8081",
        "AIGW_AUTH_MODE": "local",
    }
    for key, value in defaults.items():
        env[key] = env.get(key) or value
    expected = {key: env[key] for key in defaults}
    pi_directory = Path(env.get("PI_AGENT_DIR") or repo / "services/pi_agent")
    if not (pi_directory / "node_modules/@earendil-works/pi-ai/package.json").is_file():
        raise RuntimeError("Bundled pi-agent dependencies missing; run npm ci in services/pi_agent before restarting")
    build = implementation(pi_directory)

    def pi_ready():
        state = get(pi_base, "/ready")
        return state.get("ok") is True and state.get("implementation") == build and state.get("configuration") == expected

    try:
        matched = pi_ready()
    except (OSError, ValueError):
        matched = False
    if not matched or env.get("FORCE_PI") == "1":
        stop_owned(repo, "pi", pi_port)
        env.update(MOSS_DEPLOY_REPO=str(repo), MOSS_DEPLOY_ROLE="pi", MOSS_DEPLOY_PORT=str(pi_port))
        process = spawn([node, "service.mjs"], pi_directory, env, logs / f"pi_agent_{pi_port}.log")
        try:
            wait_ready(pi_ready, process, 30)
        except Exception:
            stop_owned(repo, "pi", pi_port)
            raise
    print(f"Memory ready: {pi_base}; 4B={json.dumps(desired)}", flush=True)


if __name__ == "__main__":
    main()
