#!/usr/bin/env python3
"""Run every formal case against one SGLang-Omni WebSocket server."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--client", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--fps", type=float)
    parser.add_argument("--input-queue-capacity", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contract = json.loads(args.contract.read_text())
    case_ids = list(contract["cases"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, case_id in enumerate(case_ids, start=1):
        output = args.output_dir / f"sglang_{case_id}_1fps.json"
        command = [
            sys.executable,
            "-u",
            str(args.client),
            "--url",
            args.url,
            "--manifest",
            str(args.manifest),
            "--case-id",
            case_id,
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--input-queue-capacity",
            str(args.input_queue_capacity),
            "--output",
            str(output),
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
