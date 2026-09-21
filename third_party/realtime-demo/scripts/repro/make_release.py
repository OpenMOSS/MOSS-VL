"""Package tracked working-tree source plus explicit delivery files, never local secrets."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(source, target, extras):
    revision = subprocess.check_output(["git", "-C", source, "rev-parse", "HEAD"], text=True).strip()
    names = set(filter(None, subprocess.check_output(["git", "-C", source, "ls-files", "-z"]).decode().split("\0")))
    for entry in extras:
        path = source / entry
        if path.is_file(): names.add(entry)
        elif path.is_dir():
            names.update(str(file.relative_to(source)) for file in path.rglob("*") if file.is_file()
                         and not {"node_modules", "__pycache__"}.intersection(file.parts))
    checksums = {}
    for name in sorted(names):
        path = source / name
        if not path.is_file(): continue
        if any(part in {".git", ".repro", "node_modules", ".venv", "logs"} for part in Path(name).parts):
            raise RuntimeError(f"Refusing runtime/private path: {name}")
        if path.name.startswith(".env") and not path.name.endswith(".example") and path.name != ".env.example":
            raise RuntimeError(f"Refusing environment file: {name}")
        if path.is_symlink(): raise RuntimeError(f"Source symlinks must be reviewed before release: {name}")
        if path.suffix in {".py", ".sh", ".json", ".mjs", ".md", ".txt", ".ts", ".tsx"}:
            if re.search(rb"(?:hf_[A-Za-z0-9]{25,}|sk-api-[A-Za-z0-9_-]{32,})", path.read_bytes()):
                raise RuntimeError(f"Possible credential in source; release aborted: {name}")
        dest = target / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        checksums[name] = digest(dest)
    manifest = {"revision": revision, "files": checksums,
                "tracked_modifications": subprocess.check_output(["git", "-C", source, "diff", "--name-only"], text=True).splitlines()}
    (target / "source-manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not (args.backend / 'pyproject.toml').is_file():
        raise RuntimeError('Backend source must contain pyproject.toml')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "moss-realtime-release"
        demo, backend = root / "demo", root / "backend"
        demo.mkdir(parents=True); backend.mkdir()
        export(ROOT, demo, ["README_zh.md", "bootstrap.sh", "services/pi_agent", "deployment/repro", "scripts/repro",
            "docs/manual_deployment.md", "docs/deployment_operations.md",
            "scripts/deploy/stop_backend.py", "scripts/deploy/memory_backend.py", "server/tests/test_repro_install.py",
            "server/tests/test_recovery_fixes.py", "scripts/tests/realtime_recovery.cjs",
            "server/tests/test_memory_batch1.py", "server/tests/test_memory_batch2.py",
            "server/tests/test_memory_shutdown.py", "server/tests/test_deploy_shutdown.py"])
        export(args.backend.resolve(), backend, ["README_zh.md", "deployment/repro",
            "deployment/moss_vl_realtime/check_env.py", "deployment/moss_vl_realtime/constraints.txt",
            "tests/unit_test/moss_vl_realtime/test_check_env.py",
            "tests/unit_test/moss_vl_realtime/test_child_cleanup.py",
            "tests/unit_test/serve/test_video_realtime_lifecycle.py"])
        (root / "README.txt").write_text("cd demo\nbash bootstrap.sh --backend-source ../backend --with-memory --with-asr --with-tts\n"
            ".venv/bin/python scripts/repro/run.py up --main-gpu 0 --memory-gpu 1\n"
            "See demo/deployment/repro/README.md for prerequisites, verification and limitations.\n")
        with tarfile.open(args.output, "w:gz", compresslevel=3) as archive:
            archive.add(root, arcname=root.name)
    value = digest(args.output)
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(f"{value}  {args.output.name}\n")
    print(json.dumps({"artifact": str(args.output), "sha256": value}))


if __name__ == "__main__": main()
