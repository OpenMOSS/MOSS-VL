"""Validation and reports for separate accuracy and latency suites."""

import json

from common import write_json
from semantic_checks import groups, performance_comparable, stats, visible


def cell(value):
    return str(value).replace("|", "\\|").replace("\r", "").replace("\n", "<br>")


def key(row):
    return row["sessions"], row["group"], row["lane"], row["case_id"]


def read_rows(output, suite, errors):
    result = {}
    for backend in ("hf", "sglang"):
        try:
            rows = json.loads((output / f"{backend}_{suite}.json").read_text())
            if not isinstance(rows, list):
                raise ValueError("Expected a list of result rows")
            result[backend] = rows
        except (OSError, ValueError) as exc:
            errors.append(f"{backend}: {exc}")
            result[backend] = []
    return result


def accuracy_report(cases, records, metadata, errors):
    expected = {
        (count, group_id, lane, case["case_id"])
        for count in metadata["sessions"]
        for group_id, group in enumerate(groups(cases, count))
        for lane, case in enumerate(group)
    }
    indexed = {}
    case_map = {c["case_id"]: c for c in cases}
    for backend, rows in records.items():
        indexed[backend] = {}
        for row in rows:
            try:
                row_key = key(row)
                case = case_map[row["case_id"]]
                valid = (
                    row_key in expected
                    and isinstance(row["text"], str)
                    and isinstance(row["token_ids"], list)
                    and bool(row["token_ids"])
                    and all(type(t) is int for t in row["token_ids"])
                    and len(row["chunks"]) == len(case["events"]) + 1
                    and all(isinstance(c, str) for c in row["chunks"])
                    and row["task_check"]["status"] in ("PASS", "REVIEW", "FAIL")
                )
                if not valid or row_key in indexed[backend]:
                    raise ValueError("Invalid or duplicate result row")
                indexed[backend][row_key] = row
                if (
                    row.get("error")
                    or row.get("capped")
                    or not row.get("input_ok")
                    or not row.get("kv_recovered", True)
                    or row["task_check"]["status"] == "FAIL"
                ):
                    errors.append(
                        f"{backend} {row_key}: execution/input/task check failed"
                    )
            except (KeyError, TypeError, ValueError, AttributeError) as exc:
                errors.append(f"{backend}: malformed result: {exc}")
        if set(indexed[backend]) != expected:
            errors.append(
                f"{backend}: incomplete matrix; expected {len(expected)} unique rows, "
                f"got {len(indexed[backend])}"
            )
    baselines = {
        b: {r["case_id"]: r for r in values.values() if r["sessions"] == 1}
        for b, values in indexed.items()
    }
    comparisons = []
    for row_key in sorted(expected):
        a, b = [indexed[name].get(row_key) for name in ("hf", "sglang")]
        if a is None or b is None:
            continue
        own = {}
        for name, row in (("hf", a), ("sglang", b)):
            one = baselines[name].get(row["case_id"])
            own[name] = bool(
                one
                and one["token_ids"] == row["token_ids"]
                and one["chunks"] == row["chunks"]
            )
        token_equal = a["token_ids"] == b["token_ids"]
        chunks_equal = a["chunks"] == b["chunks"]
        review = (
            not token_equal
            or not chunks_equal
            or not all(own.values())
            or any(r["task_check"]["status"] == "REVIEW" for r in (a, b))
        )
        comparisons.append(
            dict(
                sessions=row_key[0],
                group=row_key[1],
                lane=row_key[2],
                case_id=row_key[3],
                hf_task=a["task_check"]["status"],
                sglang_task=b["task_check"]["status"],
                token_equal=token_equal,
                text_equal=a["text"] == b["text"],
                event_text_equal=chunks_equal,
                self_single_equal=own,
                review_required=review,
            )
        )
    drift = any(
        not r["token_equal"]
        or not r["event_text_equal"]
        or not all(r["self_single_equal"].values())
        for r in comparisons
    )
    if metadata["strict_tokens"] and drift:
        errors.append("Strict token/event alignment check failed")
    review = any(r["review_required"] for r in comparisons)
    lines = [
        "# HF / SGLang 精度与输出对齐",
        "",
        "## 测试口径",
        "",
        f"- HF attention: {metadata['hf_attention']}；SGLang: FlashInfer + decode CUDA Graph。",
        f"- BF16；SGLang FP32 LM head: {metadata['sglang_fp32_lm_head']}；greedy，seed=0。",
        "- 相同素材、问题、时间戳；每次发送一个事件，生成到 silence 后发送下一事件。",
        "- HF 共享模型、独立 KV 轮转；SGLang 原生连续批处理。关闭滑窗、pooling 和 async decode。",
        "- 使用自然生成，不强制答案。每事件最多 256 token，异常/输入不完整/回收失败单独记失败。",
        "- 冻结素材用于实现回归，不是独立泛化评测；任务判据、文本相同、逐 token 相同分别报告。",
        "- 开放式描述保留 REVIEW；退出码 0 不代表人工语义审阅已完成。",
        "",
        "## 对齐汇总",
        "",
        "| Sessions | 配对样本数 | HF / SG 任务通过 | HF / SG 待审阅 | HF / SG 任务失败 | 跨后端 token 一致 | 与自身单路一致 |",
        "| ---: | ---: | --- | --- | --- | --- | --- |",
    ]
    for count in metadata["sessions"]:
        subset = [r for r in comparisons if r["sessions"] == count]
        n = len(subset)
        counts = lambda status: " / ".join(
            str(sum(r[b + "_task"] == status for r in subset)) for b in ("hf", "sglang")
        )
        own = "; ".join(
            f"{b.upper()}: {sum(r['self_single_equal'][b] for r in subset)}/{n}"
            for b in ("hf", "sglang")
        )
        lines.append(
            f"| {count} | {n} | {counts('PASS')} | {counts('REVIEW')} | {counts('FAIL')} | "
            f"{sum(r['token_equal'] for r in subset)}/{n} | {own} |"
        )
    lines.extend(
        [
            "",
            "## 逐路结果",
            "",
            "| Sessions / Group / Lane | 用例 | HF / SG 判据 | 文本一致 | Token 一致 | 事件输出一致 | HF / SG 自身单路一致 |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for r in comparisons:
        lines.append(
            f"| {r['sessions']} / {r['group']} / {r['lane']} | {cell(r['case_id'])} | "
            f"{r['hf_task']} / {r['sglang_task']} | {r['text_equal']} | {r['token_equal']} | "
            f"{r['event_text_equal']} | {r['self_single_equal']['hf']} / {r['self_single_equal']['sglang']} |"
        )
    lines.extend(["", "## 自然生成文本与事件", ""])
    for row_key in sorted(expected):
        count, group, lane, case_id = row_key
        case = case_map[case_id]
        lines.extend(
            [
                f"### {count} session / group {group} / lane {lane}: {case_id}",
                "",
                "初始问题：" + cell(case["initial_prompt"]),
                "",
                "| 事件 / 视频时间 | 问题 / 参考 | HF | SGLang |",
                "| --- | --- | --- | --- |",
            ]
        )
        pair = [indexed[b].get(row_key) for b in ("hf", "sglang")]
        for i in range(len(case["events"]) + 1):
            event = case["events"][i - 1] if i else {}
            target = (
                event.get("expected_after", "")
                if i
                else case.get("initial_expected", "")
            )
            texts = [r["chunks"][i] if r else "(missing)" for r in pair]
            if not (visible(target) or event.get("prompt") or any(texts)):
                continue
            label = f"{i - 1} / {event.get('timestamp', '-')}"
            context = str(event.get("prompt") or "") + " / " + visible(target)
            lines.append(
                "| "
                + " | ".join(cell(v or "(silence)") for v in (label, context, *texts))
                + " |"
            )
        lines.append("")
    return (
        dict(
            comparisons=comparisons,
            token_or_event_drift=drift,
            semantic_review_required=review,
            results=records,
        ),
        lines,
    )


def latency_report(cases, records, metadata, errors):
    comparable = not errors and performance_comparable(records["hf"], records["sglang"])
    comparable &= all(len(rows) == metadata["repeats"] for rows in records.values())
    try:
        import math

        comparable &= all(
            math.isfinite(r["prompt_ttft_seconds"]) and r["prompt_ttft_seconds"] > 0
            for rows in records.values()
            for r in rows
        )
    except (KeyError, TypeError):
        comparable = False
    if not comparable:
        errors.append(
            "Incomplete/invalid latency samples or mismatched prefix signatures"
        )
    speed = {}
    if comparable:
        for backend, trials in records.items():
            speed[backend] = dict(
                prefill=stats([r["prefill_seconds"] for r in trials]),
                frame=stats([v for r in trials for v in r["frame_seconds"]]),
                prompt_ttft=stats([r["prompt_ttft_seconds"] for r in trials]),
                decode=stats([v for r in trials for v in r["decode_intervals"]]),
            )
            speed[backend]["tokens_per_second"] = 1 / speed[backend]["decode"]["mean"]
    same_hardware = (
        metadata["same_gpu"]
        or len({(g["name"], g["total_mib"]) for g in metadata["gpus"]}) == 1
    )
    ratio_allowed = comparable and same_hardware
    lines = [
        "# HF / SGLang 单 Session 时延",
        "",
        "## 测试口径",
        "",
        f"- 用例：{metadata['latency_case']}；HF attention: {metadata['hf_attention']}。",
        f"- SGLang: FlashInfer + CUDA Graph；FP32 LM head: {metadata['sglang_fp32_lm_head']}。",
        f"- 同卡顺序运行：{metadata['same_gpu']}；双卡模式仍是每后端单卡，不是 TP。",
        f"- 完整预热 1 轮后计时 {metadata['repeats']} 轮；每轮 11 帧，前三帧不计入 frame 统计。",
        "- 每帧固定一个 silence；最终固定生成 64 个相同普通 token，统计后续 63 个 decode 间隔。",
        "- Initial prefill 含首个控制 token；Frame extend 含图像读取/预处理/增量执行至控制 token。",
        "- Prompt TTFT 从最终问题提交到首个原始 token，不等同于可见回答首字延迟。",
        "- 强制 token 仅用于固定工作量，不作为语义输出。计时不包含模型加载和墙钟帧等待。",
        "- SGLang 为进程内调度器；不含网络、WebSocket、Demo、文本 memory 或其他模型。",
        "- 关闭视觉滑窗/pooling/async；P95 为 nearest-rank，小样本不作为线上 SLA。",
        "- 双卡结果受设备状态影响；正式加速比优先参考同卡测试，异型号设备不计算加速比。",
        "",
        f"输入签名与采样数检查：{comparable}",
        "",
        "| 指标 | HF mean / P50 / P95 ms | SGLang mean / P50 / P95 ms | 样本数 HF / SG | HF÷SG mean |",
        "| --- | --- | --- | --- | --- |",
    ]
    if comparable:
        for name, title in (
            ("prefill", "Initial prefill"),
            ("frame", "Frame extend"),
            ("prompt_ttft", "Prompt TTFT"),
            ("decode", "TPOT"),
        ):
            a, b = speed["hf"][name], speed["sglang"][name]
            format_stats = lambda s: " / ".join(
                f"{s[k] * 1000:.3f}" for k in ("mean", "p50", "p95")
            )
            ratio = f"{a['mean'] / b['mean']:.2f}x" if ratio_allowed else "N/A"
            lines.append(
                f"| {title} | {format_stats(a)} | {format_stats(b)} | {a['count']} / {b['count']} | {ratio} |"
            )
        lines.extend(
            [
                "",
                f"Decode throughput: HF {speed['hf']['tokens_per_second']:.2f} token/s; "
                f"SGLang {speed['sglang']['tokens_per_second']:.2f} token/s.",
            ]
        )
    else:
        lines.append("| N/A | 不完整或不可比较 | 不完整或不可比较 | N/A | N/A |")
    return (
        dict(
            performance_comparable=comparable,
            ratio_allowed=ratio_allowed,
            speed=speed,
            results=records,
        ),
        lines,
    )


def render(output, cases, codes, metadata, errors=None):
    errors = list(errors or [])
    required = {"sglang"} if metadata["suite"] == "concurrency" else {"hf", "sglang"}
    if set(codes) != required or any(code != 0 for code in codes.values()):
        errors.append(f"Worker execution incomplete: {codes}")
    if metadata["suite"] == "concurrency":
        from concurrency_benchmark import report

        summary, lines = report(output, metadata, errors)
    else:
        records = read_rows(output, metadata["suite"], errors)
        builder = accuracy_report if metadata["suite"] == "accuracy" else latency_report
        summary, lines = builder(cases, records, metadata, errors)
    memory = {}
    if metadata.get("memory_monitoring"):
        for backend in required:
            try:
                measurement = json.loads(
                    (output / f"{backend}_memory.json").read_text()
                )
                peaks = [
                    measurement[k]
                    for k in (
                        "process_peak_bytes",
                        "device_peak_bytes",
                        "device_total_bytes",
                    )
                ]
                if not all(type(value) is int for value in peaks):
                    raise ValueError("Memory measurements must be integer byte counts")
                if measurement["errors"] or not 0 < peaks[0] <= peaks[1] <= peaks[2]:
                    errors.append(
                        f"{backend}: invalid or incomplete memory measurement"
                    )
                memory[backend] = measurement
            except (OSError, KeyError, TypeError, ValueError) as exc:
                errors.append(f"{backend}: missing memory measurement: {exc}")
        lines.extend(
            [
                "",
                "## 显存",
                "",
                f"mem_fraction_static={metadata.get('mem_fraction_static')}；NVML 每 50 ms 采样。",
                "80 GB 为容量参考目标，显存峰值单独记录，不作为测试通过的硬门槛。",
                "",
                "| 后端 | 进程峰值 GiB | 整卡峰值 GiB | 整卡峰值 GB |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for backend, measurement in memory.items():
            lines.append(
                f"| {backend} | {measurement['process_peak_bytes']/2**30:.3f} | "
                f"{measurement['device_peak_bytes']/2**30:.3f} | {measurement['device_peak_bytes']/10**9:.3f} |"
            )
        summary["memory_policy"] = "reference_only"
    summary["memory"] = {
        backend: {k: v for k, v in measurement.items() if k != "samples"}
        for backend, measurement in memory.items()
    }
    status = (
        "FAIL"
        if errors
        else "REVIEW" if summary.get("semantic_review_required") else "PASS"
    )
    summary.update(
        schema_version=1,
        suite=metadata["suite"],
        status=status,
        execution_or_rule_failure=bool(errors),
        metadata=metadata,
        worker_codes=codes,
        errors=errors,
    )
    lines[2:2] = [f"状态：{status}", ""]
    if errors:
        lines.extend(["", "## 未通过检查", "", *["- " + cell(e) for e in errors]])
    write_json(output / "summary.json", summary)
    (output / f"{metadata['suite']}.md").write_text("\n".join(lines) + "\n")
    return not errors
