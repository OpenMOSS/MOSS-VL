"""CPU-only tests for evaluation isolation, completeness and reporting."""

import copy
import importlib.util
import json
import sys
from contextlib import nullcontext
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[3] / "deployment/moss_vl_realtime"


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(HERE))
    result = {}
    for name in (
        "common",
        "semantic_checks",
        "evaluation_reports",
        "evaluation",
    ):
        spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        result[name] = module
    monkeypatch.setattr(
        result["evaluation"], "MemoryMonitor", lambda path: nullcontext()
    )
    return result


def fixture_data():
    cases = [
        dict(
            case_id=str(i),
            initial_prompt="Question",
            initial_expected="",
            events=[],
            contract={"mode": "manual_timeline"},
        )
        for i in range(4)
    ]
    rows = []
    for count in (1, 2, 4):
        for index, case in enumerate(cases):
            rows.append(
                dict(
                    sessions=count,
                    group=index // count,
                    lane=index % count,
                    case_id=case["case_id"],
                    text="Answer",
                    chunks=["Answer"],
                    token_ids=[7, 8],
                    task_check={"status": "PASS"},
                    input_ok=True,
                    error=None,
                    capped=False,
                    kv_recovered=True,
                )
            )
    metadata = dict(
        suite="accuracy",
        sessions=[1, 2, 4],
        strict_tokens=False,
        hf_attention="eager",
        sglang_fp32_lm_head=False,
        same_gpu=True,
        gpus=[dict(name="H200", total_mib=140000)],
        repeats=2,
        latency_case="0",
    )
    return cases, {"hf": rows, "sglang": copy.deepcopy(rows)}, metadata


def report(modules, output, cases, records, metadata, codes=None):
    for backend, rows in records.items():
        (output / f"{backend}_{metadata['suite']}.json").write_text(json.dumps(rows))
    passed = modules["evaluation_reports"].render(
        output, cases, codes if codes is not None else {"hf": 0, "sglang": 0}, metadata
    )
    summary = json.loads((output / "summary.json").read_text())
    return passed, summary


def test_full_accuracy_matrix_and_no_latency_file(modules, tmp_path):
    cases, records, metadata = fixture_data()
    passed, summary = report(modules, tmp_path, cases, records, metadata)
    assert passed and summary["status"] == "PASS"
    assert len(summary["comparisons"]) == 12
    assert all(
        r["token_equal"] and all(r["self_single_equal"].values())
        for r in summary["comparisons"]
    )
    assert (tmp_path / "accuracy.md").exists()
    assert not (tmp_path / "latency.md").exists()


def test_unthrottled_sixteen_session_capacity(modules, monkeypatch):
    from types import SimpleNamespace

    evaluation = modules["evaluation"]
    args = evaluation.parse_args(
        [
            "concurrency",
            "/model",
            "--worker",
            "sglang",
            "--sessions",
            "1",
            "2",
            "4",
            "8",
            "16",
            "--token-rate",
            "86400",
        ]
    )
    assert args.sessions == [1, 2, 4, 8, 16] and args.token_rate == 86400
    captured = {}

    def engine(path, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(close=lambda: None)

    def run(model, case, sessions, repeats, save, *, fps, token_rate):
        assert sessions == [1, 2, 4, 8, 16]
        assert repeats == 3 and fps == 1 and token_rate == 86400
        return []

    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(manual_seed=lambda seed: None)
    )
    monkeypatch.setitem(sys.modules, "semantic_engine", SimpleNamespace(Engine=engine))
    monkeypatch.setitem(
        sys.modules, "concurrency_benchmark", SimpleNamespace(run_interleaved=run)
    )
    cases, _, _ = fixture_data()
    cases[0]["case_id"] = evaluation.LATENCY_CASE
    assert evaluation.run_worker(args, cases) == 0
    assert captured["max_running_requests"] == 16


@pytest.mark.parametrize("issue", [None, "sampling", "zero", "missing", "execution"])
def test_memory_capacity_is_reference_but_real_failures_remain(
    modules, tmp_path, issue
):
    cases, records, metadata = fixture_data()
    metadata["memory_monitoring"] = True
    measurement = dict(
        process_peak_bytes=80_293_658_624,
        device_peak_bytes=81_704_648_704,
        device_total_bytes=150_754_820_096,
        budget_bytes=80_000_000_000,
        within_budget=False,
        errors=[],
    )
    for backend in records:
        value = copy.deepcopy(measurement)
        if backend == "sglang":
            if issue == "missing":
                continue
            if issue == "sampling":
                value["errors"] = ["NVML sampling failed"]
            if issue == "zero":
                value["process_peak_bytes"] = 0
        (tmp_path / f"{backend}_memory.json").write_text(json.dumps(value))
    if issue == "execution":
        records["sglang"][0]["error"] = "CUDA out of memory"
    passed, summary = report(modules, tmp_path, cases, records, metadata)
    assert passed is (issue is None)
    assert summary["status"] == ("PASS" if issue is None else "FAIL")
    assert summary["memory_policy"] == "reference_only"
    assert summary["memory"]["hf"]["device_peak_bytes"] == 81_704_648_704
    if issue is None:
        assert summary["errors"] == []
        assert "81.705" in (tmp_path / "accuracy.md").read_text()


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "duplicate",
        "bad_tokens",
        "bad_chunks",
        "rule",
        "input",
        "kv",
        "error",
        "worker",
    ],
)
def test_incomplete_or_failed_accuracy_is_not_pass(modules, tmp_path, mutation):
    cases, records, metadata = fixture_data()
    row = records["sglang"][-1]
    codes = {"hf": 0, "sglang": 0}
    if mutation == "missing":
        records["sglang"].pop()
    if mutation == "duplicate":
        records["sglang"][-1] = records["sglang"][0]
    if mutation == "bad_tokens":
        row["token_ids"] = None
    if mutation == "bad_chunks":
        row["chunks"] = []
    if mutation == "rule":
        row["task_check"]["status"] = "FAIL"
    if mutation == "input":
        row["input_ok"] = False
    if mutation == "kv":
        row["kv_recovered"] = False
    if mutation == "error":
        row["error"] = "timeout"
    if mutation == "worker":
        codes["sglang"] = 124
    passed, summary = report(modules, tmp_path, cases, records, metadata, codes)
    assert not passed and summary["status"] == "FAIL"


def test_review_is_preserved_even_for_exact_outputs(modules, tmp_path):
    cases, records, metadata = fixture_data()
    for rows in records.values():
        rows[0]["task_check"]["status"] = "REVIEW"
    passed, summary = report(modules, tmp_path, cases, records, metadata)
    assert passed and summary["status"] == "REVIEW"
    assert summary["semantic_review_required"]


@pytest.mark.parametrize("strict", [False, True])
def test_multi_session_drift_and_strict_mode(modules, tmp_path, strict):
    cases, records, metadata = fixture_data()
    records["sglang"][-1]["token_ids"] = [9]
    metadata["strict_tokens"] = strict
    passed, summary = report(modules, tmp_path, cases, records, metadata)
    assert passed is not strict
    assert summary["token_or_event_drift"]
    row = next(
        r for r in summary["comparisons"] if r["sessions"] == 4 and r["case_id"] == "3"
    )
    assert not row["token_equal"] and not row["self_single_equal"]["sglang"]


def trial():
    return dict(
        prefill_seconds=0.02,
        prompt_ttft_seconds=0.03,
        frame_seconds=[0.1] * 8,
        decode_tokens=64,
        decode_intervals=[0.01] * 63,
        signature=dict(
            forced_token=1782,
            frames=["hash"] * 11,
            encoder_tokens=2431,
            decoder_prefix_tokens=202,
        ),
    )


def test_latency_is_independent_and_aggregates_samples(modules, tmp_path):
    cases, _, metadata = fixture_data()
    metadata.update(suite="latency", sessions=[1])
    records = {b: [trial(), trial()] for b in ("hf", "sglang")}
    passed, summary = report(modules, tmp_path, cases, records, metadata)
    assert passed and summary["status"] == "PASS"
    assert summary["speed"]["hf"]["decode"]["count"] == 126
    assert summary["speed"]["hf"]["tokens_per_second"] == 100
    assert (tmp_path / "latency.md").exists()
    assert not (tmp_path / "accuracy.md").exists()


@pytest.mark.parametrize(
    "mutation", ["missing", "malformed", "ttft", "prefix", "both_prefixes", "nan"]
)
def test_latency_mismatch_never_reports_speedup(modules, tmp_path, mutation):
    cases, _, metadata = fixture_data()
    metadata.update(suite="latency", sessions=[1])
    records = {b: [trial(), trial()] for b in ("hf", "sglang")}
    if mutation == "missing":
        records["sglang"].pop()
    if mutation == "malformed":
        records["sglang"][0] = {}
    if mutation == "ttft":
        records["sglang"][0].pop("prompt_ttft_seconds")
    if mutation == "prefix":
        records["sglang"][0]["signature"]["decoder_prefix_tokens"] += 1
    if mutation == "both_prefixes":
        for rows in records.values():
            rows[1]["signature"]["decoder_prefix_tokens"] += 1
    if mutation == "nan":
        records["sglang"][0]["decode_intervals"][0] = float("nan")
    passed, summary = report(modules, tmp_path, cases, records, metadata)
    assert not passed and not summary["ratio_allowed"]
    assert not summary["speed"]


def test_distinct_gpu_models_suppress_ratio(modules, tmp_path):
    cases, _, metadata = fixture_data()
    metadata.update(
        suite="latency",
        sessions=[1],
        same_gpu=False,
        gpus=[dict(name="A100", total_mib=80000), dict(name="H200", total_mib=140000)],
    )
    records = {b: [trial(), trial()] for b in ("hf", "sglang")}
    passed, summary = report(modules, tmp_path, cases, records, metadata)
    assert passed and summary["performance_comparable"] and not summary["ratio_allowed"]


def test_invalid_json_still_produces_failure_report(modules, tmp_path):
    cases, _, metadata = fixture_data()
    (tmp_path / "hf_accuracy.json").write_text("{")
    assert not modules["evaluation_reports"].render(tmp_path, cases, {}, metadata)
    assert (tmp_path / "accuracy.md").exists()


def test_incomplete_concurrency_report_does_not_require_hf(modules, tmp_path):
    metadata = dict(suite="concurrency", sessions=[1], repeats=1)
    assert not modules["evaluation_reports"].render(
        tmp_path, [], {"sglang": 0}, metadata
    )
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["status"] == "FAIL"
    assert (tmp_path / "concurrency.md").exists()


def test_cli_and_single_gpu_waves(modules):
    evaluation = modules["evaluation"]
    args = evaluation.parse_args(["accuracy", "/model"])
    assert args.sessions == [1, 2, 4] and args.hf_attention == "eager"
    assert not args.sg_fp32_lm_head
    gpu = dict(index="2")
    assert evaluation.execution_waves([gpu]) == [[("hf", gpu)], [("sglang", gpu)]]
    assert len(evaluation.execution_waves([gpu, dict(index="3")])) == 1
    for extra in (["--sessions", "2"], ["--sessions", "1", "1"], ["--repeats", "0"]):
        with pytest.raises(SystemExit):
            evaluation.parse_args(["accuracy", "/model", *extra])
    with pytest.raises(SystemExit):
        evaluation.parse_args(["latency", "/model", "--strict-tokens"])


def test_workers_only_execute_the_requested_suite(modules, monkeypatch, tmp_path):
    from types import SimpleNamespace

    calls = []

    class FakeModel:
        def __init__(self, *args, **kwargs):
            pass

        def semantic(self, cases, count):
            calls.append(("accuracy", count))
            return [
                dict(case_id=c["case_id"], task_check={"status": "PASS"}) for c in cases
            ]

        def performance(self, case, repeats):
            calls.append(("latency", repeats))
            return [trial()] * repeats

        def close(self):
            calls.append(("close",))

    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(manual_seed=lambda seed: None)
    )
    monkeypatch.setitem(
        sys.modules, "semantic_tf", SimpleNamespace(Reference=FakeModel)
    )
    evaluation = modules["evaluation"]
    cases, _, _ = fixture_data()
    cases[0]["case_id"] = evaluation.LATENCY_CASE
    for suite in ("accuracy", "latency"):
        calls.clear()
        args = evaluation.parse_args(
            [suite, "/model", "--worker", "hf", "--output-dir", str(tmp_path)]
        )
        assert evaluation.worker(args, cases) == 0
        assert calls[-1] == ("close",)
        assert all(c[0] in (suite, "close") for c in calls)


@pytest.mark.parametrize("suite", ["accuracy", "latency"])
@pytest.mark.parametrize("timeout", [False, True])
def test_main_single_card_worker_arguments_and_cleanup(
    modules, monkeypatch, tmp_path, suite, timeout
):
    evaluation = modules["evaluation"]
    cases, records, metadata = fixture_data()
    cases[0]["case_id"] = evaluation.LATENCY_CASE
    for rows in records.values():
        for row in rows:
            if row["case_id"] == "0":
                row["case_id"] = evaluation.LATENCY_CASE
    metadata.update(suite=suite)
    output = tmp_path / "result"
    calls = []
    gpu = dict(index="2", uuid="test-gpu")
    monkeypatch.setattr(evaluation, "validate_model", lambda path: path)
    monkeypatch.setattr(evaluation, "load_suite", lambda path: cases)
    monkeypatch.setattr(evaluation, "gpu_inventory", lambda: [gpu])
    monkeypatch.setattr(evaluation, "select_gpus", lambda *args: [gpu])
    monkeypatch.setattr(evaluation, "metadata_for", lambda *args: metadata)
    monkeypatch.setattr(evaluation, "environment", lambda gpu: {})

    class FakeChild:
        def __init__(self, command, env, log):
            self.args = evaluation.parse_args(command[3:])
            calls.append(("start", self.args.worker))

        def wait(self, seconds):
            assert 0 < seconds <= 1800
            calls.append(("wait", self.args.worker))
            if timeout:
                raise TimeoutError("fixture timeout")
            rows = (
                records[self.args.worker] if suite == "accuracy" else [trial(), trial()]
            )
            (output / f"{self.args.worker}_{suite}.json").write_text(json.dumps(rows))
            return 0

        def stop(self):
            calls.append(("stop", self.args.worker))

    monkeypatch.setattr(evaluation, "Child", FakeChild)
    code = evaluation.main([suite, "/model", "--output-dir", str(output)])
    assert code == (1 if timeout else 0)
    assert calls == [
        (op, backend)
        for backend in ("hf", "sglang")
        for op in ("start", "wait", "stop")
    ]
    assert (output / f"{suite}.md").exists()
