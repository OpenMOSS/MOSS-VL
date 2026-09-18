"""Independent HF/SGLang accuracy and matched-prefix latency evaluations."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

from common import (
    CONFIG,
    HERE,
    ROOT,
    Child,
    environment,
    gpu_inventory,
    select_gpus,
    validate_model,
    write_json,
)
from evaluation_reports import render
from memory_monitor import MemoryMonitor
from semantic_checks import groups, load_suite

LATENCY_CASE = "cd067_sbpro_L2_stream_000122"


def parse_args(argv=None):
    arguments = sys.argv[1:] if argv is None else argv
    suites = ("accuracy", "latency", "concurrency")
    selected = arguments[0] if arguments and arguments[0] in suites else None
    descriptions = {
        "accuracy": "HF/SGLang controlled-input accuracy and output alignment.",
        "latency": "HF/SGLang single-session fixed-workload latency.",
        "concurrency": "SGLang independent-session latency under concurrent input.",
    }
    program = (
        f"test_{arguments[0]}.sh" if arguments and arguments[0] in suites else None
    )
    parser = argparse.ArgumentParser(
        prog=program, description=descriptions.get(selected, __doc__)
    )
    parser.add_argument("suite", choices=suites, help=argparse.SUPPRESS)
    parser.add_argument("model_path", type=Path, help="Local model directory")
    parser.add_argument("--cases-dir", type=Path, default=HERE / "cases")
    parser.add_argument(
        "--gpus",
        help=(
            "One GPU shared by all sessions"
            if selected == "concurrency"
            else "One GPU by default; optionally two independent backends"
        ),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--hf-attention",
        choices=("eager", "sdpa"),
        default="eager",
        help=(
            argparse.SUPPRESS
            if selected == "concurrency"
            else "HF attention (default: eager)"
        ),
    )
    parser.add_argument("--sg-fp32-lm-head", action="store_true")
    parser.add_argument(
        "--fps",
        type=float,
        help=(
            "Input FPS per session (default: 1)"
            if selected == "concurrency"
            else argparse.SUPPRESS
        ),
    )
    parser.add_argument(
        "--token-rate",
        type=float,
        help=(
            "Output rate per session (default: 10 tokens/s; use 86400 for unthrottled testing)"
            if selected == "concurrency"
            else argparse.SUPPRESS
        ),
    )
    parser.add_argument(
        "--sessions",
        nargs="+",
        type=int,
        help=(
            "Session counts (default: 1 2 4)"
            if selected == "accuracy"
            else (
                "Session counts: 1 2 4 8 16 (default: 1 2 4 8)"
                if selected == "concurrency"
                else argparse.SUPPRESS
            )
        ),
    )
    parser.add_argument(
        "--strict-tokens",
        action="store_true",
        help=(
            "Fail on token/event drift" if selected == "accuracy" else argparse.SUPPRESS
        ),
    )
    parser.add_argument(
        "--repeats",
        type=int,
        help=(
            "Measured repetitions (default: 5)"
            if selected == "latency"
            else (
                "Measured repetitions (default: 3)"
                if selected == "concurrency"
                else argparse.SUPPRESS
            )
        ),
    )
    parser.add_argument(
        "--timeout", type=int, default=1800, help="Worker deadline in seconds"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker", choices=("hf", "sglang"), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.suite == "accuracy" and args.repeats is not None:
        parser.error("--repeats applies only to latency and concurrency tests")
    if args.suite == "latency" and args.sessions is not None:
        args.suite = "concurrency"
    if args.suite == "concurrency" and args.hf_attention != "eager":
        parser.error(
            "--hf-attention does not apply to the SGLang-only concurrency test"
        )
    if args.suite != "accuracy" and args.strict_tokens:
        parser.error("--strict-tokens applies only to accuracy")
    if args.suite != "concurrency" and (
        args.fps is not None or args.token_rate is not None
    ):
        parser.error("--fps and --token-rate require the concurrency test")
    args.fps = args.fps if args.fps is not None else 1.0
    args.token_rate = args.token_rate if args.token_rate is not None else 10.0
    if any(not math.isfinite(v) or v <= 0 for v in (args.fps, args.token_rate)):
        parser.error("--fps and --token-rate must be finite positive values")
    args.sessions = (
        args.sessions
        if args.sessions is not None
        else (
            [1, 2, 4]
            if args.suite == "accuracy"
            else [1, 2, 4, 8] if args.suite == "concurrency" else [1]
        )
    )
    args.repeats = (
        args.repeats
        if args.repeats is not None
        else (3 if args.suite == "concurrency" else 5)
    )
    if (
        1 not in args.sessions
        or any(
            n not in ((1, 2, 4, 8, 16) if args.suite == "concurrency" else (1, 2, 4))
            for n in args.sessions
        )
        or len(set(args.sessions)) != len(args.sessions)
    ):
        parser.error("--sessions must contain distinct supported counts including 1")
    if args.repeats < 1 or args.timeout < 1:
        parser.error("--repeats and --timeout must be positive")
    args.sessions = sorted(args.sessions)
    return args


def execution_waves(gpus):
    if len(gpus) == 1:
        return [[("hf", gpus[0])], [("sglang", gpus[0])]]
    if len(gpus) == 2:
        return [[("hf", gpus[0]), ("sglang", gpus[1])]]
    raise ValueError("Use one GPU or two GPUs, not tensor parallelism")


def worker(args, cases):
    with MemoryMonitor(args.output_dir / f"{args.worker}_memory.json"):
        return run_worker(args, cases)


def run_worker(args, cases):
    import torch

    random.seed(0)
    torch.manual_seed(0)
    if args.suite == "concurrency" and args.worker != "sglang":
        raise ValueError("Concurrency measures the native SGLang scheduler only")
    if args.worker == "hf":
        from semantic_tf import Reference

        model = Reference(str(args.model_path), attention=args.hf_attention)
    else:
        from semantic_engine import Engine

        # NPU: allow pinning the eval engine to a spare card and a context
        # that fits one card's KV pool; CUDA keeps the historical defaults.
        eval_device = os.environ.get("MOSS_EVAL_DEVICE")
        eval_context = int(os.environ.get("MOSS_EVAL_CONTEXT", "131072"))
        model = Engine(
            args.model_path,
            device=eval_device,
            context_length=eval_context,
            disable_cuda_graph=bool(os.environ.get("MOSS_EVAL_DISABLE_GRAPH")),
            max_running_requests=(
                max(8, max(args.sessions)) if args.suite == "concurrency" else 4
            ),
            server_args_overrides={
                "prefill_attention_backend": "flashinfer",
                "decode_attention_backend": "flashinfer",
                "enable_fp32_lm_head": args.sg_fp32_lm_head,
            },
        )
    rows = []
    try:
        if args.suite == "accuracy":
            for count in args.sessions:
                for group_id, group in enumerate(groups(cases, count)):
                    current = model.semantic(group, count)
                    for row in current:
                        row["group"] = group_id
                    rows.extend(current)
                    write_json(args.output_dir / f"{args.worker}_accuracy.json", rows)
                    print(
                        "GROUP",
                        args.worker,
                        count,
                        group_id,
                        [(r["case_id"], r["task_check"]) for r in current],
                        flush=True,
                    )
        elif args.suite == "concurrency":
            from concurrency_benchmark import run_interleaved

            case = next(c for c in cases if c["case_id"] == LATENCY_CASE)
            rows = run_interleaved(
                model,
                case,
                args.sessions,
                args.repeats,
                lambda data: write_json(
                    args.output_dir / "sglang_concurrency.json", data
                ),
                fps=args.fps,
                token_rate=args.token_rate,
            )
        else:
            case = next(c for c in cases if c["case_id"] == LATENCY_CASE)
            rows = model.performance(case, args.repeats)
            write_json(args.output_dir / f"{args.worker}_latency.json", rows)
    finally:
        model.close()
    return 0


def metadata_for(args, cases, gpus):
    versions = {}
    for name in ("torch", "transformers", "sglang", "flashinfer-python"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return dict(
        schema_version=1,
        suite=args.suite,
        created_at=datetime.now(timezone.utc).isoformat(),
        model_path=str(args.model_path),
        model_config_sha256=hashlib.sha256(
            (args.model_path / "config.json").read_bytes()
        ).hexdigest(),
        model_sources={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in args.model_path.glob("*.py")
        },
        gpus=gpus,
        same_gpu=len(gpus) == 1,
        versions=versions,
        repository_commit=revision.stdout.strip(),
        seed=0,
        dtype="bfloat16",
        mem_fraction_static=CONFIG["mem_fraction_static"],
        memory_monitoring=True,
        hf_attention=args.hf_attention,
        sglang_attention="flashinfer",
        sglang_fp32_lm_head=args.sg_fp32_lm_head,
        cuda_graph=True,
        async_decode=False,
        visual_window=False,
        pooling=False,
        schedule=(
            "independent open-loop senders"
            if args.suite == "concurrency"
            else "one event after silence"
        ),
        fps_wall_clock=args.suite == "concurrency",
        concurrency_protocol=(
            "independent_realtime_v1" if args.suite == "concurrency" else None
        ),
        concurrency_fps=args.fps if args.suite == "concurrency" else None,
        concurrency_token_rate=args.token_rate if args.suite == "concurrency" else None,
        timing_units="seconds",
        sessions=args.sessions,
        test_max_running_requests=(
            max(8, max(args.sessions)) if args.suite == "concurrency" else 4
        ),
        strict_tokens=args.strict_tokens,
        repeats=args.repeats,
        warmup_trials=0 if args.suite == "concurrency" else 1,
        warmup_frames=2 if args.suite == "concurrency" else 3,
        concurrency_order=(
            "rotating interleaved trials" if args.suite == "concurrency" else None
        ),
        latency_case=LATENCY_CASE,
        case_ids=[c["case_id"] for c in cases],
        sources={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in HERE.glob("*.py")
        },
        inputs={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                args.cases_dir / "manifest.jsonl",
                args.cases_dir / "formal_eval_contract.json",
            )
        },
    )


def main(argv=None):
    args = parse_args(argv)
    args.model_path = validate_model(args.model_path)
    args.cases_dir = args.cases_dir.expanduser().resolve()
    cases = load_suite(args.cases_dir)
    if args.suite != "accuracy" and not any(
        c["case_id"] == LATENCY_CASE for c in cases
    ):
        raise ValueError(f"Latency suite requires {LATENCY_CASE}")
    if args.worker:
        return worker(args, cases)
    if args.dry_run:
        print(
            json.dumps(
                dict(
                    suite=args.suite,
                    model_path=str(args.model_path),
                    mem_fraction_static=CONFIG["mem_fraction_static"],
                    cases=[c["case_id"] for c in cases],
                    gpus=args.gpus or "auto: one idle GPU",
                    sessions=args.sessions,
                    hf_attention=args.hf_attention,
                    sg_fp32_lm_head=args.sg_fp32_lm_head,
                    repeats=args.repeats if args.suite != "accuracy" else None,
                    fps=args.fps if args.suite == "concurrency" else None,
                    token_rate=args.token_rate if args.suite == "concurrency" else None,
                ),
                indent=2,
            )
        )
        return 0
    gpus = select_gpus(
        args.gpus, gpu_inventory(), os.environ.get("CUDA_VISIBLE_DEVICES")
    )
    if args.suite == "concurrency":
        if len(gpus) != 1:
            raise ValueError(
                "Concurrent sessions must share one GPU; select one --gpus value"
            )
        waves = [[("sglang", gpus[0])]]
    else:
        waves = execution_waves(gpus)
    output = (
        args.output_dir
        or ROOT
        / "results"
        / args.suite
        / (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "-"
            + uuid.uuid4().hex[:6]
        )
    ).resolve()
    output.mkdir(parents=True, exist_ok=False)
    metadata = metadata_for(args, cases, gpus)
    write_json(output / "metadata.json", metadata)
    write_json(output / "cases.json", cases)
    print("OUTPUT", output, flush=True)
    children, codes, errors = [], {}, []
    try:
        for wave in waves:
            active = []
            for backend, gpu in wave:
                cmd = [
                    sys.executable,
                    "-u",
                    str(Path(__file__).resolve()),
                    args.suite,
                    str(args.model_path),
                    "--cases-dir",
                    str(args.cases_dir),
                    "--output-dir",
                    str(output),
                    "--worker",
                    backend,
                    "--hf-attention",
                    args.hf_attention,
                ]
                if args.suite in ("accuracy", "concurrency"):
                    cmd.extend(["--sessions", *map(str, args.sessions)])
                if args.suite != "accuracy":
                    cmd.extend(["--repeats", str(args.repeats)])
                if args.suite == "concurrency":
                    cmd.extend(
                        ["--fps", str(args.fps), "--token-rate", str(args.token_rate)]
                    )
                if args.sg_fp32_lm_head:
                    cmd.append("--sg-fp32-lm-head")
                env = environment(gpu["uuid"])
                env["REALTIME_FRAME_WINDOW_ENABLED"] = "0"
                child = Child(cmd, env, output / f"{backend}.log")
                children.append(child)
                active.append((backend, child, time.monotonic() + args.timeout))
                print("START", backend, "GPU", gpu["index"], flush=True)
            for backend, child, deadline in active:
                try:
                    codes[backend] = child.wait(max(0, deadline - time.monotonic()))
                except TimeoutError as exc:
                    codes[backend] = 124
                    errors.append(str(exc))
                finally:
                    child.stop()
                    children.remove(child)
                print("DONE", backend, codes[backend], flush=True)
    except BaseException as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        for child in reversed(children):
            child.stop()
        passed = render(output, cases, codes, metadata, errors)
        print("REPORT", output / f"{args.suite}.md", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":

    def stop(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
