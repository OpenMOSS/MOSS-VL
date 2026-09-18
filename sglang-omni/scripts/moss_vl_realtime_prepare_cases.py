#!/usr/bin/env python3
"""Select and materialize five canonical 1-fps realtime training cases."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sglang_omni.models.moss_vl_realtime.benchmark_cases import (
    compile_training_record,
    materialize_case_frames,
)


@dataclass(frozen=True)
class SourceSpec:
    source_tag: str
    task_shape: str
    path: Path


SOURCE_DEFINITIONS = (
    (
        "cd067_sbpro_L2_stream",
        "step trigger with one short response",
        Path("feeder_sbpro/reanchored.clean_v1.jsonl"),
    ),
    (
        "cd067_pvqabase_L2_magqa",
        "answer evolution as evidence changes",
        Path("output_cd052_pvqa_feeder/" "L3_magqa_answer_evolution.clean_v1.jsonl"),
    ),
    (
        "cd067_pvqatv_L2_inject",
        "streaming question answering with subtitle injection",
        Path(
            "output_cd052_tvqa6_ceilfix/"
            "tv_TVQA_injection_answer_silence.clean_v1.jsonl"
        ),
    ),
    (
        "cd067_omnimmipa_L2_pa",
        "proactive event or action description",
        Path("output_cd050_omnimmi_pa/L1_PA_omnimmi_token_silence.clean_v1.jsonl"),
    ),
    (
        "cd067_ovorec_L2_slow1fps",
        "progressive 1-fps counting and recognition",
        Path("output_cd049_rec_slow1fps_v2/REC_slow1fps_train.clean_v1.jsonl"),
    ),
)


def source_specs(data_root: Path) -> tuple[SourceSpec, ...]:
    return tuple(
        SourceSpec(source_tag, task_shape, data_root / relative_path)
        for source_tag, task_shape, relative_path in SOURCE_DEFINITIONS
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Root containing the five realtime QA source subdirectories.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-frames", type=int, default=20)
    parser.add_argument("--max-frames", type=int, default=60)
    parser.add_argument("--scan-limit", type=int, default=5000)
    parser.add_argument("--candidates-per-source", type=int, default=100)
    return parser.parse_args()


def _has_meaningful_answer(case: dict[str, Any]) -> bool:
    text = case["initial_expected"] + "".join(
        str(event["expected_after"]) for event in case["events"]
    )
    text = re.sub(r"<\|[^|]+\|>", "", text)
    return bool(text.strip())


def select_case(spec: SourceSpec, args: argparse.Namespace) -> dict[str, Any]:
    candidates: list[tuple[tuple[int, int, int], dict[str, Any]]] = []
    with spec.path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if args.scan_limit > 0 and line_number > args.scan_limit:
                break
            try:
                record = json.loads(line)
                case = compile_training_record(
                    record,
                    case_id=f"{spec.source_tag}_{line_number:06d}",
                    source_tag=spec.source_tag,
                    task_shape=spec.task_shape,
                    source_path=str(spec.path),
                    source_line=line_number,
                )
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
            frame_count = int(case["frame_count"])
            if case["frame_interval_seconds"] != 1.0:
                continue
            if frame_count > args.max_frames or not _has_meaningful_answer(case):
                continue
            score = (abs(frame_count - args.target_frames), frame_count, line_number)
            candidates.append((score, case))
            candidates.sort(key=lambda item: item[0])
            del candidates[args.candidates_per_source :]

    for _, case in candidates:
        if Path(case["video_path"]).is_file():
            return case
    raise RuntimeError(
        f"no readable <= {args.max_frames}-frame 1-fps case found for {spec.source_tag}"
    )


def main() -> None:
    args = parse_args()
    if args.target_frames <= 0 or args.max_frames <= 0:
        raise ValueError("frame limits must be positive")
    if args.target_frames > args.max_frames:
        raise ValueError("--target-frames must not exceed --max-frames")
    if args.candidates_per_source <= 0:
        raise ValueError("--candidates-per-source must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = [select_case(spec, args) for spec in source_specs(args.data_root)]
    for case in cases:
        materialize_case_frames(case, args.output_dir)
        print(
            f"selected {case['case_id']}: {case['frame_count']} frames, "
            f"video={case['video_path']}",
            flush=True,
        )

    manifest_path = args.output_dir / "manifest.jsonl"
    temporary_path = manifest_path.with_suffix(".jsonl.tmp")
    with temporary_path.open("w", encoding="utf-8") as output:
        for case in cases:
            output.write(json.dumps(case, ensure_ascii=False) + "\n")
    temporary_path.replace(manifest_path)
    print(f"wrote {len(cases)} cases to {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
