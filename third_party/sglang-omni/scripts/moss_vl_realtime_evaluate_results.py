#!/usr/bin/env python3
"""Evaluate formal MOSS-VL realtime results against one frozen contract."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any

CONTROL_TOKEN = re.compile(r"<\|[^>]+\|>")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    return parser.parse_args()


def visible_text(text: str) -> str:
    return CONTROL_TOKEN.sub("", text).strip()


def load_results(paths: list[Path]) -> list[dict[str, Any]]:
    results = []
    for path in paths:
        payload = json.loads(path.read_text())
        records = payload.get("results") if isinstance(payload, dict) else None
        if records is None:
            records = [payload]
        for record in records:
            record = dict(record)
            record["result_path"] = str(path)
            results.append(record)
    return results


def expected_outputs(case: dict[str, Any]) -> list[str]:
    outputs = []
    initial = visible_text(str(case.get("initial_expected", "")))
    if initial:
        outputs.append(initial)
    for event in case["events"]:
        text = visible_text(str(event.get("expected_after", "")))
        if text:
            outputs.append(text)
    return outputs


def result_text(result: dict[str, Any]) -> str:
    if result.get("backend") == "sglang-omni":
        return visible_text(
            str(result.get("normalized_text") or "")
            or "".join(
                str(event.get("delta", "")) for event in result.get("events", [])
            )
        )
    return visible_text(
        str(result.get("raw_text") or result.get("normalized_text") or "")
    )


def first_content_seconds(result: dict[str, Any]) -> float | None:
    records = (
        result.get("events", [])
        if result.get("backend") == "sglang-omni"
        else result.get("outputs", [])
    )
    for record in records:
        text = str(record.get("delta", record.get("text", "")))
        if visible_text(text):
            value = record.get("elapsed_seconds")
            return None if value is None else float(value)
    return None


def processed_latencies(result: dict[str, Any]) -> list[dict[str, Any]]:
    if result.get("backend") != "sglang-omni":
        return []
    accepted = {
        int(event["seq_no"]): float(event["elapsed_seconds"])
        for event in result.get("events", [])
        if event.get("type") in ("input.frame.accepted", "input.prompt.accepted")
    }
    records = []
    for event in result.get("events", []):
        event_type = event.get("type")
        if event_type not in ("input.frame.processed", "input.prompt.processed"):
            continue
        seq_no = int(event["seq_no"])
        records.append(
            {
                "case_id": result["case_id"],
                "seq_no": seq_no,
                "type": "frame" if event_type == "input.frame.processed" else "prompt",
                "seconds": float(event["elapsed_seconds"]) - accepted[seq_no],
            }
        )
    return records


def latency_stats(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not records:
        return None
    values = sorted(float(record["seconds"]) for record in records)

    def percentile(fraction: float) -> float:
        index = min(len(values) - 1, int((len(values) - 1) * fraction))
        return values[index]

    return {
        "count": len(values),
        "p50_seconds": statistics.median(values),
        "p95_seconds": percentile(0.95),
        "max_seconds": values[-1],
    }


def transport_checks(case: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    expected_frames = int(case["frame_count"])
    expected_prompts = sum(event["type"] == "prompt" for event in case["events"])
    if result.get("backend") == "sglang-omni":
        events = result.get("events", [])
        frame_count = sum(
            event.get("type") == "input.frame.processed" for event in events
        )
        prompt_count = sum(
            event.get("type") == "input.prompt.processed" for event in events
        )
        dropped = 0
    else:
        events = result.get("input_events", [])
        frame_count = sum(event.get("type") == "frame" for event in events)
        prompt_count = sum(event.get("type") == "prompt" for event in events)
        dropped = sum(bool(event.get("dropped")) for event in events)
    return {
        "expected_frames": expected_frames,
        "observed_frames": frame_count,
        "expected_prompts": expected_prompts,
        "observed_prompts": prompt_count,
        "dropped_frames": dropped,
        "passed": (
            frame_count == expected_frames
            and prompt_count == expected_prompts
            and dropped == 0
        ),
    }


def content_check(text: str, contract: dict[str, Any]) -> dict[str, Any]:
    mode = contract["mode"]
    if mode == "manual_timeline":
        return {"mode": mode, "passed": None, "missing": []}
    if mode == "exact":
        expected = str(contract["expected_text"])
        return {
            "mode": mode,
            "passed": text.strip() == expected,
            "expected": expected,
            "missing": [] if text.strip() == expected else [expected],
        }
    required = [str(item) for item in contract.get("required_text", [])]
    missing = [item for item in required if item.casefold() not in text.casefold()]
    return {"mode": mode, "passed": not missing, "missing": missing}


def main() -> None:
    args = parse_args()
    manifest = {
        case["case_id"]: case
        for line in args.manifest.read_text().splitlines()
        if line
        for case in [json.loads(line)]
    }
    contract = json.loads(args.contract.read_text())
    summaries = []
    for result in load_results(args.result):
        case_id = str(result["case_id"])
        case = manifest[case_id]
        text = result_text(result)
        transport = transport_checks(case, result)
        content = content_check(text, contract["cases"][case_id])
        if not transport["passed"] or content["passed"] is False:
            status = "fail"
        elif content["passed"] is None:
            status = "needs_review"
        else:
            status = "pass"
        summaries.append(
            {
                "backend": result["backend"],
                "case_id": case_id,
                "status": status,
                "elapsed_seconds": result.get("elapsed_seconds"),
                "first_content_seconds": first_content_seconds(result),
                "visible_text": text,
                "expected_outputs": expected_outputs(case),
                "transport": transport,
                "content": content,
                "processed_latencies": processed_latencies(result),
                "result_path": result["result_path"],
            }
        )

    all_latency_records = [
        record for summary in summaries for record in summary["processed_latencies"]
    ]
    frame_latencies = [
        record for record in all_latency_records if record["type"] == "frame"
    ]
    prompt_latencies = [
        record for record in all_latency_records if record["type"] == "prompt"
    ]
    coldest = (
        max(frame_latencies, key=lambda record: record["seconds"])
        if frame_latencies
        else None
    )
    warm_frame_latencies = [
        record for record in frame_latencies if coldest is None or record is not coldest
    ]
    profiling = {
        "sglang_frame_all": latency_stats(frame_latencies),
        "sglang_frame_warm_excluding_coldest": latency_stats(warm_frame_latencies),
        "sglang_prompt": latency_stats(prompt_latencies),
        "coldest_event": coldest,
    }
    payload = {
        "schema_version": 2,
        "contract": str(args.contract),
        "summaries": summaries,
        "profiling": profiling,
        "backend_performance_comparison": {
            "comparable": False,
            "reason": (
                "The functional 1 FPS traces do not use a shared prefill/decode "
                "timing contract or equal completion-token counts."
            ),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    lines = [
        "# MOSS-VL Realtime Formal 1 FPS Matrix",
        "",
        "| Backend | Case | Status | Frames | Prompts | Dropped | First visible trace |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for item in summaries:
        transport = item["transport"]
        first = item["first_content_seconds"]
        lines.append(
            "| {backend} | {case} | {status} | {frames}/{expected_frames} | "
            "{prompts}/{expected_prompts} | {dropped} | {first} |".format(
                backend=item["backend"],
                case=item["case_id"],
                status=item["status"],
                frames=transport["observed_frames"],
                expected_frames=transport["expected_frames"],
                prompts=transport["observed_prompts"],
                expected_prompts=transport["expected_prompts"],
                dropped=transport["dropped_frames"],
                first="-" if first is None else f"{first:.3f}s",
            )
        )
    lines.extend(
        [
            "",
            "## Backend Performance Comparison",
            "",
            (
                "Withheld: these functional traces do not use a shared prefill/decode "
                "timing contract or equal completion-token counts. Session duration and "
                "first-visible-trace timestamps must not be interpreted as a Transformers "
                "versus SGLang performance comparison."
            ),
        ]
    )
    lines.extend(["", "## Outputs", ""])
    for item in summaries:
        lines.append(f"### {item['backend']} / {item['case_id']}")
        lines.append("")
        lines.append(item["visible_text"] or "(no visible text)")
        lines.append("")
    if coldest is not None:
        all_stats = profiling["sglang_frame_all"]
        warm_stats = profiling["sglang_frame_warm_excluding_coldest"]
        prompt_stats = profiling["sglang_prompt"]
        lines.extend(
            [
                "## SGLang Processed Latency",
                "",
                (
                    f"- All frames: n={all_stats['count']}, "
                    f"p50={all_stats['p50_seconds']:.4f}s, "
                    f"p95={all_stats['p95_seconds']:.4f}s, "
                    f"max={all_stats['max_seconds']:.4f}s"
                ),
                (
                    f"- Warm frames excluding the single coldest event: "
                    f"n={warm_stats['count']}, p50={warm_stats['p50_seconds']:.4f}s, "
                    f"p95={warm_stats['p95_seconds']:.4f}s, "
                    f"max={warm_stats['max_seconds']:.4f}s"
                ),
                (
                    "- Prompt-only: none"
                    if prompt_stats is None
                    else f"- Prompt-only: n={prompt_stats['count']}, max={prompt_stats['max_seconds']:.4f}s"
                ),
                (
                    f"- Coldest event: {coldest['case_id']} seq={coldest['seq_no']} "
                    f"({coldest['seconds']:.4f}s)"
                ),
                "",
            ]
        )
    args.output_markdown.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
