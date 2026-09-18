"""Frozen-case loading and explicit semantic/performance comparison rules."""

import hashlib
import json
import math
import re
import statistics
from pathlib import Path

CONTROL = re.compile(r"<\|[^>]+\|>")


def visible(text):
    return CONTROL.sub("", text).strip()


def load_suite(directory):
    directory = Path(directory).resolve()
    contract = json.loads((directory / "formal_eval_contract.json").read_text())
    cases = []
    identifiers = set()
    for line in (directory / "manifest.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        source = json.loads(line)
        if source["case_id"] in identifiers:
            raise ValueError("Duplicate case identifier")
        identifiers.add(source["case_id"])
        case = {
            k: source[k]
            for k in ("case_id", "system_prompt", "initial_prompt", "frame_count")
        }
        case["task_shape"] = source.get("task_shape", "")
        case["initial_expected"] = source.get("initial_expected", "")
        case["contract"] = contract["cases"][case["case_id"]]
        case["events"] = []
        timestamp = 0.0
        for i, event in enumerate(source["events"]):
            if event["type"] not in ("frame", "prompt"):
                raise ValueError("Unsupported event type")
            if event["seq_no"] != i:
                raise ValueError("Case sequence numbers must be contiguous")
            item = {
                k: event[k]
                for k in ("type", "seq_no", "prompt", "expected_after")
                if k in event
            }
            previous_timestamp = timestamp
            timestamp = float(event.get("timestamp", timestamp))
            if not math.isfinite(timestamp) or timestamp < previous_timestamp:
                raise ValueError("Timestamps must be finite and nondecreasing")
            item["timestamp"] = timestamp
            if event["type"] == "frame":
                path = Path(event["frame_path"])
                if not path.is_absolute():
                    path = directory / path
                path = path.resolve()
                if not path.is_file():
                    raise ValueError(f"Missing frame: {path}")
                item["frame_path"] = str(path)
                item["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                if event.get("sha256") and event["sha256"] != item["sha256"]:
                    raise ValueError(f"Frame checksum mismatch: {path.name}")
            case["events"].append(item)
        if sum(e["type"] == "frame" for e in case["events"]) != case["frame_count"]:
            raise ValueError("Frame count does not match manifest")
        cases.append(case)
    if not cases:
        raise ValueError("The semantic suite is empty")
    return cases


def groups(cases, count):
    if not cases or count < 1:
        raise ValueError("Use nonempty cases and positive session count")
    return [
        [cases[(start + i) % len(cases)] for i in range(count)]
        for start in range(0, len(cases), count)
    ]


def assess(text, rule, *, error=None, capped=False):
    text = visible(text)
    if error or capped:
        return {"status": "FAIL", "reason": error or "per-event token cap reached"}
    if not text:
        return {"status": "FAIL", "reason": "no visible answer"}
    if rule["mode"] == "manual_timeline":
        return {
            "status": "REVIEW",
            "reason": "open-ended timeline needs semantic review",
        }
    if rule["mode"] == "exact":
        passed = text == rule["expected_text"].strip()
        return {"status": "PASS" if passed else "FAIL", "reason": "exact answer"}
    missing = [s for s in rule["required_text"] if s.casefold() not in text.casefold()]
    return {
        "status": "FAIL" if missing else "PASS",
        "reason": (
            "missing: " + repr(missing) if missing else "required content matched"
        ),
    }


def stats(samples):
    values = sorted(samples)
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "p50": statistics.median(values),
        "p95": values[math.ceil(0.95 * len(values)) - 1],
    }


def performance_comparable(tf, sg):
    try:
        return (
            bool(tf)
            and len(tf) == len(sg)
            and all(
                not a.get("error")
                and not b.get("error")
                and a["decode_tokens"] == b["decode_tokens"] == 64
                and len(a["decode_intervals"]) == len(b["decode_intervals"]) == 63
                and len(a["frame_seconds"]) == len(b["frame_seconds"]) == 8
                and all(
                    math.isfinite(t) and t > 0
                    for r in (a, b)
                    for t in r["decode_intervals"]
                    + r["frame_seconds"]
                    + [r["prefill_seconds"]]
                )
                and a["signature"] == b["signature"] == tf[0]["signature"]
                and set(a["signature"])
                == {"forced_token", "frames", "encoder_tokens", "decoder_prefix_tokens"}
                and len(a["signature"]["frames"]) == 11
                and a["signature"]["encoder_tokens"] > 0
                and a["signature"]["decoder_prefix_tokens"] > 0
                for a, b in zip(tf, sg)
            )
        )
    except (KeyError, TypeError, ValueError, AttributeError):
        return False
