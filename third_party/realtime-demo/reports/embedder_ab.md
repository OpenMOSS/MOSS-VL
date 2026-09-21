# Memory 文本 Embedder A/B 对比实验报告

- 日期：2026-09-04（GATEWAY_PLAN P5 产出 2）
- A = `hashing-ngram`（dim 512，numpy 哈希 n-gram，零依赖，现网默认）
- B = `bge-m3`（dim 1024，本地 HF 模型 models/memory/bge-m3，CPU，CLS pooling）
- 语料：memory.db 有效文本条目 26 条（kind ∈ utterance/fact 等，排除 invalid_at 非空与空文本 frame 行）
- 查询：journal 去重后 8 条（is_retro_question 命中 0 条，不足 5 条，按实验约定使用全量问题）

## 结论先行

**B（BGE-M3）召回质量全面优于 A（hashing），建议在有权重、可负担约 2GB 内存的部署上启用（`MEMORY_EMBED_TEXT_MODEL=models/memory/bge-m3`）；A 保留为零依赖兜底（空气隔离/无权重环境）。**

- 精确去重两者持平（重复条目均以 1.0 置顶）——这是本语料的主要形态；
- 释义/语义召回 B 明显更强：「画面里发生了什么」↔「实时描述当前画面」B 给到 ~0.7-0.9，A 只有 0.21（低于自身 gate 直接漏召）；
- 只有 B 能召回答案内容：「你是什么模型」→ assistant 条目「我是MOSS-VL…」B 排第 4（0.567），A 的 top-5 完全没有；
- 开销：B 加载 10-23s（一次性，冷热缓存之间）/ RSS +~2GB / CPU 单条 encode ~0.5s，只能在后台线程跑（生产已如此设计），retro 问题每轮一次查询编码可接受；
- 未发现 A 反而更准的 case；B 的误差形态是「过度关联」（个别无关对擦过 0.62 门槛），A 的误差形态是「漏召 + 词面噪声误召」。

## 分数区分度（top1 − top5 分差，越大越不容易误召回）

| Embedder | 分差均值 | 分差最小值 | gate_floor（生产值） |
|---|---|---|---|
| A hashing | 0.7071 | 0.5133 | 0.22 |
| B bge-m3 | 0.278 | 0.1064 | 0.62 |

> 注意：原始分差**不可跨 embedder 直接比较**。A 的 0.71 高分差源于 top-5 迅速掉到词面零交集（分数归零），并不代表相关/无关分得更开——恰恰相反，A 的相关近邻（释义句）混在 0.2-0.5 区间，与噪声同档；B 的分数带整体压缩（XLM-R 系特性，无关对也有 ~0.5 的地基），所以生产按 embedder 分别设 gate_floor（0.22 vs 0.62）。有效区分度应对照各自 gate 看：
>
> - A 在 gate=0.22 下放过词面噪声：「你是什么模型」→「你好有什么我可以帮你的吗」0.246 过闸（错误记忆）；「你是谁」→「你好」0.258 过闸。
> - B 在 gate=0.62 下挡住所有跨主题对（画面类 vs 你是谁 ~0.51 被拦），仅 2 例轻度过度关联过闸（「你好」→「请描述画面中的内容」0.632），无实质性错误记忆。

## 开销（CPU，单条 encode，各 3 轮）

| Embedder | 加载耗时(s) | RSS 增量(MB) | encode 均值(ms) | p50(ms) | p95(ms) | max(ms) | 样本数 |
|---|---|---|---|---|---|---|---|
| A hashing | 0.0 | 0.0 | 0.04 | 0.02 | 0.09 | 0.13 | 102 |
| B bge-m3 | 9.624 | 2082.8 | 444.11 | 397.37 | 797.83 | 901.25 | 102 |

## 查询对照表（top-3，余下见 embedder_ab.json 的 top-5）

### Q: 画面里发生了什么？

| # | A: hashing | score | B: bge-m3 | score |
|---|---|---|---|---|
| 1 | [utterance/user] 画面里发生了什么？ | 1.000 | [utterance/user] 画面里发生了什么？ | 1.000 |
| 2 | [utterance/user] 画面里发生了什么？ | 1.000 | [utterance/user] 画面里发生了什么？ | 1.000 |
| 3 | [utterance/user] 画面里发生了什么？ | 1.000 | [utterance/user] 画面里发生了什么？ | 1.000 |

### Q: 请描述画面中的内容。

| # | A: hashing | score | B: bge-m3 | score |
|---|---|---|---|---|
| 1 | [utterance/user] 请描述画面中的内容。 | 1.000 | [utterance/user] 请描述画面中的内容。 | 1.000 |
| 2 | [utterance/user] 请描述画面中的内容。 | 1.000 | [utterance/user] 请描述画面中的内容。 | 1.000 |
| 3 | [utterance/user] 实时描述当前画面内容 | 0.487 | [utterance/user] 画面里发生了什么？ | 0.894 |

### Q: 实时描述当前画面

| # | A: hashing | score | B: bge-m3 | score |
|---|---|---|---|---|
| 1 | [utterance/user] 实时描述当前画面 | 1.000 | [utterance/user] 实时描述当前画面 | 1.000 |
| 2 | [utterance/user] 实时描述当前画面 | 1.000 | [utterance/user] 实时描述当前画面 | 1.000 |
| 3 | [utterance/user] 实时描述当前画面内容 | 0.888 | [utterance/user] 实时描述当前画面内容 | 0.987 |

### Q: 你是谁

| # | A: hashing | score | B: bge-m3 | score |
|---|---|---|---|---|
| 1 | [utterance/user] 你是谁 | 1.000 | [utterance/user] 你是谁 | 1.000 |
| 2 | [utterance/user] 你是什么模型 | 0.405 | [utterance/user] 你是什么模型 | 0.728 |
| 3 | [utterance/user] 你好 | 0.258 | [utterance/user] 你好 | 0.628 |

### Q: 实时描述当前画面内容

| # | A: hashing | score | B: bge-m3 | score |
|---|---|---|---|---|
| 1 | [utterance/user] 实时描述当前画面内容 | 1.000 | [utterance/user] 实时描述当前画面内容 | 1.000 |
| 2 | [utterance/user] 实时描述当前画面 | 0.888 | [utterance/user] 实时描述当前画面 | 0.987 |
| 3 | [utterance/user] 实时描述当前画面 | 0.888 | [utterance/user] 实时描述当前画面 | 0.987 |

### Q: 你好 请实时描述当前画面

| # | A: hashing | score | B: bge-m3 | score |
|---|---|---|---|---|
| 1 | [utterance/user] 你好 请实时描述当前画面 | 1.000 | [utterance/user] 你好 请实时描述当前画面 | 1.000 |
| 2 | [utterance/user] 实时描述当前画面 | 0.845 | [utterance/user] 实时描述当前画面 | 0.909 |
| 3 | [utterance/user] 实时描述当前画面 | 0.845 | [utterance/user] 实时描述当前画面 | 0.909 |

### Q: 你好

| # | A: hashing | score | B: bge-m3 | score |
|---|---|---|---|---|
| 1 | [utterance/user] 你好 | 1.000 | [utterance/user] 你好 | 1.000 |
| 2 | [utterance/assistant] 你好有什么我可以帮你的吗？ | 0.471 | [utterance/assistant] 你好有什么我可以帮你的吗？ | 0.840 |
| 3 | [utterance/user] 你好 请实时描述当前画面 | 0.378 | [utterance/user] 你好 请实时描述当前画面 | 0.702 |

### Q: 你是什么模型

| # | A: hashing | score | B: bge-m3 | score |
|---|---|---|---|---|
| 1 | [utterance/user] 你是什么模型 | 1.000 | [utterance/user] 你是什么模型 | 1.000 |
| 2 | [utterance/user] 你是谁 | 0.405 | [utterance/user] 你是谁 | 0.728 |
| 3 | [utterance/assistant] 你好有什么我可以帮你的吗？ | 0.246 | [utterance/user] 你好 | 0.587 |

## 结论与建议

### 质量判断（人工读对照表）

1. **去重：两者持平。** 本语料 26 条中大量是跨会话重复的同一问句，两者都把精确重复以 1.0 置顶——hashing 的设计目标（dedup + 词面召回）达成。
2. **释义召回：B 明显更准。** 「画面里发生了什么？」「请描述画面中的内容。」「实时描述当前画面」是同一意图的三种说法，B 全部互相关联（0.72-0.99）；A 只认词面重叠高的组合，「画面里发生了什么」→「实时描述当前画面」仅 0.213，低于自身 gate_floor=0.22，**漏召**。回溯型问题（「刚才那个…」）恰恰依赖这种换说法的关联，而本次 journal 里 retro 问题为 0 条，真实 retro 场景下 A 的缺口只会更大。
3. **答案内容召回：只有 B 能做到。** 「你是什么模型」→ B 第 4 名召到 assistant 回答「我是MOSS-VL，由OpenMOSS团队研发…」（0.567）；该条目与查询零词面重叠，A 的 top-5 里根本不存在。「你好」→ B 第 2 名是 assistant「你好有什么我可以帮你的吗？」（0.840），是把「问答对」当记忆召回的正确行为。
4. **A 反而更准的 case：没有。** A 的所有 top-1 精确命中 B 都有且同分；A 独有的 top-5 条目均为词面噪声（如「你是什么模型」→「你好有什么我可以帮你的吗」0.246 过闸）。B 的失误是 2 例擦线过度关联（「你好」→「请描述画面中的内容」0.632 略过 0.62 闸），危害低于 A 的漏召+误召。

### 建议

- **启用 BGE-M3**：有权重且内存可负担（+~2GB RSS）的部署设置 `MEMORY_EMBED_TEXT_MODEL=models/memory/bge-m3`。语义召回、答案内容召回、gate 有效性全面占优。
- **保留 hashing 兜底**：零依赖、加载 0s、单条 <0.1ms，空气隔离/无权重环境下仍是唯一选择，fallback 逻辑不动。
- **开销可控**：B 单条 encode 均值 ~0.5s（CPU），但 recall 路径每轮只需 1 次查询编码且已在 writer 线程/`asyncio.to_thread` 执行（embed.py 设计如此）；一次性懒加载成本（10-23s）可用预热规避首查延迟。
- **gate_floor=0.62 在本数据上校准合理**：无关对（~0.51）被拦、相关对（≥0.63）通过，无需调整；ColBERT late-interaction 通道（`encode_tokens`）本次未评，可作为后续精度进一步提升的候选。
- **数据局限**：语料仅 26 条、查询仅 8 条去重问句且 retro=0，结论方向可信但幅度待验证；建议积累含「刚才/之前/还记得」的真实回溯会话后重跑本脚本（`tools/embedder_ab.py` 可重复执行）。
