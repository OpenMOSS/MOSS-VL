#!/usr/bin/env python3
"""Run server regressions in isolated processes, including legacy script suites."""
import argparse
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
LEGACY = {"test_offline_sglang", "test_session_manager", "test_persistence", "test_video_source",
          "test_session_ws", "test_vlm_workers"}


def clean_environment():
    generator = runpy.run_path(str(ROOT / "scripts/dev/check_env.py"))
    names = {name for _, name in generator["scan_config"]() + generator["scan_direct_readers"]()}
    names.update(generator["SHELL_ONLY"])
    names.update(generator["INTERNAL"])
    names.update(["http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"])
    env = {key: value for key, value in os.environ.items() if key not in names}
    env.update(ENV_DEPLOY_FILE="", CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1",
               PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               NO_PROXY="*", no_proxy="*")
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()
    output = (args.output_dir or Path(tempfile.mkdtemp(prefix="moss-demo-tests-"))).resolve()
    output.mkdir(parents=True, exist_ok=True)
    env = clean_environment()
    results = []
    for path in sorted((ROOT / "server/tests").glob("test_*.py")):
        if path.stem in LEGACY:
            command = [sys.executable, "-m", f"server.tests.{path.stem}"]
        else:
            command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(path),
                       f"--junitxml={output / (path.stem + '.xml')}"]
        log = output / (path.stem + ".log")
        try:
            with log.open("w") as handle:
                run = subprocess.run(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT,
                                     timeout=args.timeout)
            code = run.returncode
        except subprocess.TimeoutExpired:
            code = 124
        results.append({"suite": path.stem, "exit_code": code, "legacy": path.stem in LEGACY})
        print(f"{'PASS' if code == 0 else 'FAIL'} {path.stem}", flush=True)
        if code:
            print("\n".join(log.read_text(errors="replace").splitlines()[-45:]), flush=True)
    (output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    failed = sum(row["exit_code"] != 0 for row in results)
    print(f"{len(results) - failed}/{len(results)} suites passed; logs: {output}", flush=True)
    return int(failed != 0)


if __name__ == "__main__":
    raise SystemExit(main())
