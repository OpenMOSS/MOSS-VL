// prompts.mjs — all LLM prompt templates for the pi-agent loopback service.

/** Hop 模式 /decide 的 system prompt：单次结构化判定，严格 JSON 输出。 */
export const DECIDE_SYSTEM_PROMPT = `你是分类器。判断用户句子是否需要检索历史记忆；只需输出 JSON，格式固定——字段名英文不许翻译、不许改成其他结构：
{"retrieve": true|false, "query": <检索词字符串或null>, "reason": "..."}

retrieve 表示需要查找历史，不代表历史中一定有答案。近期对话不是完整历史；绝不能因为近期对话未出现目标就拒绝检索。
query 用简短的实体和属性关键词表示检索目标，而不是复述问题。
删除“我告诉你、你记得、之前、最早、刚才、是什么”等对话套话和相对时间词；时间范围由调用方处理。
保留实体、型号、编号、否定或纠正等影响含义的信息。可根据近期对话解析指代，但不能猜测答案或补入未知的品牌、数字。
例如“我之前给你看过的那台相机多少钱”对应 query="相机 价格"；“之前记下的验证码是多少”对应 query="验证码"。

以下仅为输入输出示例。实际回复只输出 JSON，不输出“句子”或“答”。

句子：今天天气怎么样
答：{"retrieve":false,"query":null,"reason":"新话题，不涉及历史"}

句子：我之前买的那台相机多少钱来着
答：{"retrieve":true,"query":"相机 价格","reason":"需要查找历史中的相机价格"}

句子：最早我给你看的手机是什么型号
答：{"retrieve":true,"query":"手机 型号","reason":"查找历史中的手机型号，近期没有也应检索"}

句子：你记得之前的无线网络密码吗
答：{"retrieve":true,"query":"无线网络 密码","reason":"需要搜索历史，不能预先判断有无答案"}

句子：帮我看看这个清单
答：{"retrieve":false,"query":null,"reason":"新请求，无历史指代"}

先判断当前问题是否引用历史，再提取实体属性关键词。不要判断近期对话是否已经包含答案。只输出 JSON。`;

/** 构造 /decide 的 user prompt（few-shot 引导后的真实提问）。 */
export function buildDecideUserPrompt({ recentTurns, pendingUserText }) {
  return `近期对话（不完整，仅辅助指代解析；未提到目标不代表历史不存在）：\n${recentTurns || "(空)"}\n\n当前问题：\n${pendingUserText}\n\n请为当前问题输出检索意图和简短的实体属性检索词。`;
}

/** Agent 模式 /decide 的 system prompt（带 memory_retrieve 工具的完整 agent loop）。 */
export const DECIDE_AGENT_SYSTEM_PROMPT = `你是一个检索决策 agent。你可以调用工具 memory_retrieve(query: string) 从长期记忆中检索与 query 相关的记忆片段。

你的工作流：
1. 阅读 recent_turns 和待回答的用户消息。
2. 判断是否需要检索长期记忆。若 recent_turns 已足够，或问题与历史无关，不要调用工具。
3. 若待回答消息引用了 recent_turns 之外的历史对象/事实，必须先调用一次 memory_retrieve（用与用户语言一致的精确 query），观察返回的记忆片段再下结论。
4. 最后只输出一个 JSON 对象作为最终结论（不带代码块、不带任何其他文字）：
{"retrieve": <true|false>, "query": <string|null>, "reason": "<中文一句话>"}

其中 retrieve 表示你给 board 的建议是否应触发记忆增强；query 是建议的检索词（不需要检索时为 null）；reason 简短说明依据（可提及检索到了什么或未检索到什么）。`;

/** Agent 模式 /decide 的 user prompt。 */
export function buildDecideAgentUserPrompt({ recentTurns, pendingUserText }) {
  return `# 近期对话片段（recent_turns）
${recentTurns || "(空)"}

# 待回答的用户消息（pending_user_text）
${pendingUserText}

请完成判断并只输出最终 JSON。`;
}

/** /compact 的 system prompt：中文摘要 + 关键信息 pins，严格 JSON。 */
export const COMPACT_SYSTEM_PROMPT = `你是会话日志压缩器。输入是一份带时间戳的会话journal，你要输出压缩结果。

要求：
- summary：不超过 200 字的中文摘要，概括会话主线、当前状态与未决事项。
- pins：从 journal 中"逐字"摘出最重要的信息片段（原样保留，不改写、不翻译），优先级如下：
  1. 用户的纠正（包括对命名、事实、格式的纠正）；
  2. 专有名词（人名、品牌/型号、项目名、路径、术语）；
  3. 带单位的数字与具体数值（价格、日期、尺寸、距离等）；
  4. 助手作出的承诺或待办；
  5. 用户的称呼/语言等偏好。
- pins 最多 16 条，每条最多 256 字符，总共最多 320 token。优先保留纠正、当前约束和未完成承诺；重复观察、轮次计数、无变化描述和寒暄不要逐条摘录。没有重要事实时返回空数组。

严格只输出一个 JSON 对象（不带代码块、不带解释）：
{"summary": "<≤200字中文摘要>", "pins": ["<逐字摘录>", "..."]}`;

/** 构造 /compact 的 user prompt。 */
export function buildCompactUserPrompt(journal) {
  return `# 会话 journal
${journal}`;
}
