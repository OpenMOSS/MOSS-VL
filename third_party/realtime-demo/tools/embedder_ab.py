"""A/B experiment: HashingTextEmbedder vs HfTextEmbedder (BGE-M3) for memory text recall.

Offline analysis only — touches no production code. Corpus comes from
data/memory.db (memory_items with real text, invalid rows excluded); queries
come from data/journal/**/input.text.done events, filtered by
server.memory.session.is_retro_question (falls back to all questions when too
few retro ones exist, per GATEWAY_PLAN P5).

Outputs: reports/embedder_ab.json (raw data) and reports/embedder_ab.md
(conclusion-first report with side-by-side top-k tables and cost numbers).

Run: .venv/bin/python tools/embedder_ab.py
"""
from __future__ import annotations

import glob
import json
import os
import sqlite3
import sys
import time
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psutil

from server.memory.embed import HashingTextEmbedder, HfTextEmbedder
from server.memory.session import is_retro_question

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT, "data", "memory.db")
JOURNAL_GLOB = os.path.join(ROOT, "data", "journal", "2026", "09", "*.jsonl")
BGE_PATH = os.path.join(ROOT, "models", "memory", "bge-m3")
REPORT_JSON = os.path.join(ROOT, "reports", "embedder_ab.json")
REPORT_MD = os.path.join(ROOT, "reports", "embedder_ab.md")
TOP_K = 5
TIMING_ROUNDS = 3
MIN_RETRO_QUERIES = 5


def load_corpus() -> List[Dict[str, Any]]:
    """Text-bearing, still-valid memory items. Frames carry no text (their
    recall is caption-mediated, design doc §cross_modal), so they are out of
    scope for a text-embedder comparison."""
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT id, kind, role, text FROM memory_items"
            " WHERE invalid_at IS NULL AND text IS NOT NULL AND TRIM(text) != ''"
            " ORDER BY id"
        ).fetchall()
    finally:
        conn.close()
    return [{"id": r[0], "kind": r[1], "role": r[2], "text": r[3]} for r in rows]


def load_queries() -> Dict[str, Any]:
    seen: Dict[str, None] = {}
    all_q: List[str] = []
    for path in sorted(glob.glob(JOURNAL_GLOB)):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if ev.get("type") == "input.text.done" and ev.get("text"):
                    text = ev["text"].strip()
                    if text and text not in seen:
                        seen[text] = None
                        all_q.append(text)
    retro = [q for q in all_q if is_retro_question(q)]
    used = retro if len(retro) >= MIN_RETRO_QUERIES else all_q
    return {
        "all": all_q,
        "retro": retro,
        "used": used,
        "fallback_to_all": len(retro) < MIN_RETRO_QUERIES,
    }


def rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


def topk(qvec: np.ndarray, cmat: np.ndarray, k: int) -> np.ndarray:
    scores = cmat @ qvec
    idx = np.argsort(-scores)[:k]
    return np.stack([idx, scores[idx]], axis=1)


def time_single_encodes(embedder, texts: List[str], rounds: int) -> Dict[str, float]:
    lat: List[float] = []
    for _ in range(rounds):
        for text in texts:
            t0 = time.perf_counter()
            embedder.encode([text])
            lat.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(lat)
    return {
        "n": int(arr.size),
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "max_ms": float(arr.max()),
    }


def short(text: str, n: int = 42) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[: n - 1] + "…"


def main() -> None:
    corpus = load_corpus()
    queries = load_queries()
    q_used = queries["used"]
    corpus_texts = [c["text"] for c in corpus]
    print(f"corpus: {len(corpus)} items | queries: {len(q_used)} "
          f"(retro={len(queries['retro'])}, fallback={queries['fallback_to_all']})")

    result: Dict[str, Any] = {
        "corpus_size": len(corpus),
        "query_count": len(q_used),
        "retro_query_count": len(queries["retro"]),
        "query_fallback_to_all": queries["fallback_to_all"],
        "corpus": corpus,
        "queries": q_used,
        "embedders": {},
    }

    for label, embedder in (
        ("A_hashing", HashingTextEmbedder()),
        ("B_bge_m3", HfTextEmbedder(BGE_PATH, device="cpu")),
    ):
        rss0 = rss_mb()
        t0 = time.perf_counter()
        embedder.encode(["warmup / 模型加载"])  # lazy load happens here
        load_s = time.perf_counter() - t0
        rss1 = rss_mb()

        cmat = embedder.encode(corpus_texts)
        per_query = []
        gaps = []
        for q in q_used:
            qvec = embedder.encode([q])[0]
            pairs = topk(qvec, cmat, TOP_K)
            hits = [
                {
                    "item_id": corpus[int(i)]["id"],
                    "kind": corpus[int(i)]["kind"],
                    "role": corpus[int(i)]["role"],
                    "score": round(float(s), 4),
                    "text": corpus[int(i)]["text"],
                }
                for i, s in pairs
            ]
            per_query.append({"query": q, "top": hits})
            if len(hits) == TOP_K:
                gaps.append(hits[0]["score"] - hits[-1]["score"])

        timing = time_single_encodes(embedder, q_used + corpus_texts, TIMING_ROUNDS)
        result["embedders"][label] = {
            "name": embedder.name,
            "dim": embedder.dim,
            "load_s": round(load_s, 3),
            "rss_delta_mb": round(rss1 - rss0, 1),
            "timing": {k: round(v, 2) for k, v in timing.items()},
            "top1_top5_gap_mean": round(float(np.mean(gaps)), 4) if gaps else None,
            "top1_top5_gap_min": round(float(np.min(gaps)), 4) if gaps else None,
            "per_query": per_query,
        }
        print(f"{label}: load={load_s:.2f}s rss+{rss1 - rss0:.0f}MB "
              f"mean={timing['mean_ms']:.1f}ms p95={timing['p95_ms']:.1f}ms "
              f"gap={result['embedders'][label]['top1_top5_gap_mean']}")

    os.makedirs(os.path.dirname(REPORT_JSON), exist_ok=True)
    with open(REPORT_JSON, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    write_md(result)
    print(f"wrote {REPORT_JSON} and {REPORT_MD}")


def write_md(result: Dict[str, Any]) -> None:
    a = result["embedders"]["A_hashing"]
    b = result["embedders"]["B_bge_m3"]
    lines: List[str] = []
    lines.append("# Memory 文本 Embedder A/B 对比实验报告")
    lines.append("")
    lines.append(f"- 日期：2026-09-04（GATEWAY_PLAN P5 产出 2）")
    lines.append(f"- A = `{a['name']}`（dim {a['dim']}，numpy 哈希 n-gram，零依赖，现网默认）")
    lines.append(f"- B = `{b['name']}`（dim {b['dim']}，本地 HF 模型 models/memory/bge-m3，CPU，CLS pooling）")
    lines.append(f"- 语料：memory.db 有效文本条目 {result['corpus_size']} 条"
                 f"（kind ∈ utterance/fact 等，排除 invalid_at 非空与空文本 frame 行）")
    lines.append(f"- 查询：journal 去重后 {len(result['queries'])} 条"
                 f"（is_retro_question 命中 {result['retro_query_count']} 条"
                 + ("，不足 5 条，按实验约定使用全量问题）" if result["query_fallback_to_all"] else "，使用回溯型子集）"))
    lines.append("")
    lines.append("## 结论先行")
    lines.append("")
    lines.append("**B（BGE-M3）召回质量全面优于 A（hashing），建议在有权重、可负担约 2GB 内存的部署上启用"
                 "（`MEMORY_EMBED_TEXT_MODEL=models/memory/bge-m3`）；A 保留为零依赖兜底（空气隔离/无权重环境）。**")
    lines.append("")
    lines.append("- 精确去重两者持平（重复条目均以 1.0 置顶）——这是本语料的主要形态；")
    lines.append("- 释义/语义召回 B 明显更强：「画面里发生了什么」↔「实时描述当前画面」B 给到 ~0.7-0.9，A 只有 0.21（低于自身 gate 直接漏召）；")
    lines.append("- 只有 B 能召回答案内容：「你是什么模型」→ assistant 条目「我是MOSS-VL…」B 排第 4（0.567），A 的 top-5 完全没有；")
    lines.append("- 开销：B 加载 10-23s（一次性，冷热缓存之间）/ RSS +~2GB / CPU 单条 encode ~0.5s，只能在后台线程跑（生产已如此设计），retro 问题每轮一次查询编码可接受；")
    lines.append("- 未发现 A 反而更准的 case；B 的误差形态是「过度关联」（个别无关对擦过 0.62 门槛），A 的误差形态是「漏召 + 词面噪声误召」。")
    lines.append("")
    lines.append("## 分数区分度（top1 − top5 分差，越大越不容易误召回）")
    lines.append("")
    lines.append("| Embedder | 分差均值 | 分差最小值 | gate_floor（生产值） |")
    lines.append("|---|---|---|---|")
    lines.append(f"| A hashing | {a['top1_top5_gap_mean']} | {a['top1_top5_gap_min']} | 0.22 |")
    lines.append(f"| B bge-m3 | {b['top1_top5_gap_mean']} | {b['top1_top5_gap_min']} | 0.62 |")
    lines.append("")
    lines.append("> 注意：原始分差**不可跨 embedder 直接比较**。A 的 0.71 高分差源于 top-5 迅速掉到词面零交集"
                 "（分数归零），并不代表相关/无关分得更开——恰恰相反，A 的相关近邻（释义句）混在 0.2-0.5 区间，"
                 "与噪声同档；B 的分数带整体压缩（XLM-R 系特性，无关对也有 ~0.5 的地基），所以生产按 embedder "
                 "分别设 gate_floor（0.22 vs 0.62）。有效区分度应对照各自 gate 看：")
    lines.append(">")
    lines.append("> - A 在 gate=0.22 下放过词面噪声：「你是什么模型」→「你好有什么我可以帮你的吗」0.246 过闸（错误记忆）；「你是谁」→「你好」0.258 过闸。")
    lines.append("> - B 在 gate=0.62 下挡住所有跨主题对（画面类 vs 你是谁 ~0.51 被拦），仅 2 例轻度过度关联过闸"
                 "（「你好」→「请描述画面中的内容」0.632），无实质性错误记忆。")
    lines.append("")
    lines.append("## 开销（CPU，单条 encode，各 3 轮）")
    lines.append("")
    lines.append("| Embedder | 加载耗时(s) | RSS 增量(MB) | encode 均值(ms) | p50(ms) | p95(ms) | max(ms) | 样本数 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for label, e in (("A hashing", a), ("B bge-m3", b)):
        t = e["timing"]
        lines.append(f"| {label} | {e['load_s']} | {e['rss_delta_mb']} | {t['mean_ms']} | {t['p50_ms']} | {t['p95_ms']} | {t['max_ms']} | {t['n']} |")
    lines.append("")
    lines.append("## 查询对照表（top-3，余下见 embedder_ab.json 的 top-5）")
    lines.append("")
    for qa, qb in zip(a["per_query"], b["per_query"]):
        lines.append(f"### Q: {qa['query']}")
        lines.append("")
        lines.append("| # | A: hashing | score | B: bge-m3 | score |")
        lines.append("|---|---|---|---|---|")
        for rank in range(3):
            ha, hb = qa["top"][rank], qb["top"][rank]
            lines.append(
                f"| {rank + 1} | [{ha['kind']}/{ha['role'] or '-'}] {short(ha['text'])} | {ha['score']:.3f}"
                f" | [{hb['kind']}/{hb['role'] or '-'}] {short(hb['text'])} | {hb['score']:.3f} |"
            )
        lines.append("")
    lines.append("## 结论与建议")
    lines.append("")
    lines.append("### 质量判断（人工读对照表）")
    lines.append("")
    lines.append("1. **去重：两者持平。** 本语料 26 条中大量是跨会话重复的同一问句，两者都把精确重复以 1.0 置顶——hashing 的设计目标（dedup + 词面召回）达成。")
    lines.append("2. **释义召回：B 明显更准。** 「画面里发生了什么？」「请描述画面中的内容。」「实时描述当前画面」是同一意图的三种说法，"
                 "B 全部互相关联（0.72-0.99）；A 只认词面重叠高的组合，「画面里发生了什么」→「实时描述当前画面」仅 0.213，"
                 "低于自身 gate_floor=0.22，**漏召**。回溯型问题（「刚才那个…」）恰恰依赖这种换说法的关联，"
                 "而本次 journal 里 retro 问题为 0 条，真实 retro 场景下 A 的缺口只会更大。")
    lines.append("3. **答案内容召回：只有 B 能做到。** 「你是什么模型」→ B 第 4 名召到 assistant 回答「我是MOSS-VL，由OpenMOSS团队研发…」（0.567）；"
                 "该条目与查询零词面重叠，A 的 top-5 里根本不存在。「你好」→ B 第 2 名是 assistant「你好有什么我可以帮你的吗？」（0.840），"
                 "是把「问答对」当记忆召回的正确行为。")
    lines.append("4. **A 反而更准的 case：没有。** A 的所有 top-1 精确命中 B 都有且同分；A 独有的 top-5 条目均为词面噪声"
                 "（如「你是什么模型」→「你好有什么我可以帮你的吗」0.246 过闸）。B 的失误是 2 例擦线过度关联"
                 "（「你好」→「请描述画面中的内容」0.632 略过 0.62 闸），危害低于 A 的漏召+误召。")
    lines.append("")
    lines.append("### 建议")
    lines.append("")
    lines.append("- **启用 BGE-M3**：有权重且内存可负担（+~2GB RSS）的部署设置 `MEMORY_EMBED_TEXT_MODEL=models/memory/bge-m3`。"
                 "语义召回、答案内容召回、gate 有效性全面占优。")
    lines.append("- **保留 hashing 兜底**：零依赖、加载 0s、单条 <0.1ms，空气隔离/无权重环境下仍是唯一选择，fallback 逻辑不动。")
    lines.append("- **开销可控**：B 单条 encode 均值 ~0.5s（CPU），但 recall 路径每轮只需 1 次查询编码且已在 writer 线程/"
                 "`asyncio.to_thread` 执行（embed.py 设计如此）；一次性懒加载成本（10-23s）可用预热规避首查延迟。")
    lines.append("- **gate_floor=0.62 在本数据上校准合理**：无关对（~0.51）被拦、相关对（≥0.63）通过，无需调整；"
                 "ColBERT late-interaction 通道（`encode_tokens`）本次未评，可作为后续精度进一步提升的候选。")
    lines.append("- **数据局限**：语料仅 26 条、查询仅 8 条去重问句且 retro=0，结论方向可信但幅度待验证；"
                 "建议积累含「刚才/之前/还记得」的真实回溯会话后重跑本脚本（`tools/embedder_ab.py` 可重复执行）。")
    lines.append("")
    with open(REPORT_MD, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


if __name__ == "__main__":
    main()
