#!/usr/bin/env python3
"""Run every formal case in a fresh Transformers subprocess."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--final-wait", type=float, default=3.0)
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--fps", type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contract = json.loads(args.contract.read_text())
    case_ids = list(contract["cases"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, case_id in enumerate(case_ids, start=1):
        output = args.output_dir / f"transformers_{case_id}_1fps.json"
        command = [
            sys.executable,
            "-u",
            str(args.runner),
            "--model-path",
            args.model_path,
            "--manifest",
            str(args.manifest),
            "--case-id",
            case_id,
            "--output",
            str(output),
            "--device",
            "cuda:0",
            "--attn-implementation",
            args.attn_implementation,
            "--final-wait",
            str(args.final_wait),
        ]
        if args.fps is not None:
            command.extend(["--fps", str(args.fps)])
        print(f"[{index}/{len(case_ids)}] {case_id}", flush=True)
        subprocess.run(command, check=True)
        payload = json.loads(output.read_text())
        if payload.get("error"):
            raise RuntimeError(f"{case_id} failed: {payload['error']}")
    print(f"wrote {len(case_ids)} results to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
