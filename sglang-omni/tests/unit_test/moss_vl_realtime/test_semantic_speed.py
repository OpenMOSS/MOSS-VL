"""Semantic criteria, frozen inputs and matched-token benchmark checks."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[3] / "deployment/moss_vl_realtime"


@pytest.fixture
def modules(monkeypatch):
    result = {}
    for name in ("common", "semantic_checks"):
        spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        result[name] = module
    return result


@pytest.mark.parametrize(
    "answer,status",
    [("0", "PASS"), (" 0 ", "PASS"), ("00", "FAIL"), ("1", "FAIL"), ("", "FAIL")],
)
def test_exact_count_is_not_nonempty_check(modules, answer, status):
    check = modules["semantic_checks"].assess(
        answer, {"mode": "exact", "expected_text": "0"}
    )
    assert check["status"] == status


def test_open_description_requires_review_and_caps_fail(modules):
    assess = modules["semantic_checks"].assess
    rule = {"mode": "manual_timeline"}
    assert assess("Players pass the ball.", rule)["status"] == "REVIEW"
    assert assess("<|silence|>", rule)["status"] == "FAIL"
    assert assess("Players pass the ball.", rule, capped=True)["status"] == "FAIL"


def test_content_contract_is_explicit(modules):
    assess = modules["semantic_checks"].assess
    rule = {"mode": "contains_all", "required_text": ["keep the change"]}
    assert assess("Keep the change.", rule)["status"] == "PASS"
    assert assess("She paid for food.", rule)["status"] == "FAIL"
    assert assess("Keep the change.", rule, error="timeout")["status"] == "FAIL"


def test_groups_cover_all_five_cases(modules):
    groups = modules["semantic_checks"].groups
    for count in (1, 2, 4):
        gs = groups(list(range(5)), count)
        assert all(len(g) == count for g in gs)
        assert set(x for g in gs for x in g) == set(range(5))


def test_standard_suite_has_four_cases_and_seventy_seven_frames(modules):
    cases = modules["semantic_checks"].load_suite(HERE / "cases")
    assert len(cases) == 4
    assert sum(c["frame_count"] for c in cases) == 77
    assert not any("ovorec" in c["case_id"] for c in cases)
    assert not any("pvqabase" in c["case_id"] for c in cases)
    assert all(c["contract"]["mode"] != "manual_timeline" for c in cases)
    stapler = next(c for c in cases if c["case_id"] == "cd067_sbpro_L2_stream_000530")
    assert stapler["frame_count"] == 9
    assert stapler["contract"] == {
        "mode": "exact",
        "expected_text": "Pull open the stapler",
    }
    assert (
        modules["semantic_checks"].assess(
            "Pull open the stapler" * 2, stapler["contract"]
        )["status"]
        == "FAIL"
    )
    for count in (1, 2, 4):
        flattened = [
            c["case_id"]
            for g in modules["semantic_checks"].groups(cases, count)
            for c in g
        ]
        assert len(flattened) == len(set(flattened)) == 4


def test_percentiles_use_nearest_rank(modules):
    stats = modules["semantic_checks"].stats
    assert stats([1, 2, 9])["p95"] == 9
    assert stats([1, 2, 9])["p50"] == 2
    assert stats([])["mean"] is None


def trial():
    return dict(
        decode_tokens=64,
        decode_intervals=[0.01] * 63,
        frame_seconds=[0.1] * 8,
        prefill_seconds=0.02,
        signature=dict(
            forced_token=1782,
            frames=["hash"] * 11,
            encoder_tokens=2431,
            decoder_prefix_tokens=202,
        ),
    )


@pytest.mark.parametrize(
    "mutation", ["token_count", "visible_deltas", "frames", "prefix", "nan"]
)
def test_mismatched_work_does_not_get_a_speedup(modules, mutation):
    compare = modules["semantic_checks"].performance_comparable
    a, b = trial(), trial()
    assert compare([a], [b])
    if mutation == "token_count":
        b["decode_tokens"] = 63
    if mutation == "visible_deltas":
        b["decode_intervals"].pop()
    if mutation == "frames":
        b["frame_seconds"].pop()
    if mutation == "prefix":
        b["signature"]["decoder_prefix_tokens"] += 1
    if mutation == "nan":
        b["decode_intervals"][0] = float("nan")
    assert not compare([a], [b])


def test_frame_loading_preserves_timestamps_and_checks_hashes(modules, tmp_path):
    frame = tmp_path / "f.png"
    frame.write_bytes(b"fixture-png-bytes")
    case = dict(
        case_id="case",
        system_prompt="System",
        initial_prompt="Question",
        frame_count=1,
        initial_expected="<|silence|>",
        events=[
            dict(
                type="frame",
                seq_no=0,
                timestamp=192.0,
                frame_path="f.png",
                sha256=hashlib.sha256(frame.read_bytes()).hexdigest(),
                expected_after="Answer",
            ),
            dict(type="prompt", seq_no=1, prompt="Question"),
        ],
    )
    (tmp_path / "manifest.jsonl").write_text(json.dumps(case) + "\n")
    (tmp_path / "formal_eval_contract.json").write_text(
        json.dumps({"cases": {"case": {"mode": "manual_timeline"}}})
    )
    loaded = modules["semantic_checks"].load_suite(tmp_path)
    assert loaded[0]["events"][0]["frame_path"] == str(frame)
    assert loaded[0]["events"][0]["timestamp"] == 192.0
    assert loaded[0]["events"][1]["timestamp"] == 192.0
    frame.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        modules["semantic_checks"].load_suite(tmp_path)
