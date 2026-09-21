// decide.mjs — POST /decide 的实现，双模式：
//   hop   （默认）单次结构化 LLM 调用，坏 JSON 重试一次；
//   agent  基于 @earendil-works/pi-agent-core Agent 的完整 agent loop，
//          注册 memory_retrieve 工具回调 board 的记忆检索接口。

import { Agent } from "@earendil-works/pi-agent-core";
import { budgetHook } from "./budget.mjs";
import { decisionSchema } from "./schemas.mjs";
import { requestSignal } from "./deadline.mjs";
import { Type, streamSimple } from "@earendil-works/pi-ai";
import { buildModel, completeJson, getApiKey, messageText, parseJsonStrict } from "./aigw.mjs";
import {
  DECIDE_SYSTEM_PROMPT,
  buildDecideUserPrompt,
  DECIDE_AGENT_SYSTEM_PROMPT,
  buildDecideAgentUserPrompt,
} from "./prompts.mjs";

export const DEFAULT_BOARD_MEMORY_URL = "http://127.0.0.1:8081";

function decideModel() {
  // decide 可以走本地小模型（AIGW_DECIDE_BASE_URL 指到自建端点时生效），
  // compact 仍用 AIGW 的大模型拿摘要质量。
  return buildModel(
    process.env.AIGW_DECIDE_MODEL || process.env.AIGW_MODEL || "Qwen3-4B-Instruct-2507",
    process.env.AIGW_DECIDE_BASE_URL || undefined,
  );
}

function decideTimeoutMs() {
  return Number(process.env.PI_DECIDE_TIMEOUT_MS || 20000);
}

/** 把 LLM 输出规整成契约形状并做类型校验。 */
export function normalizeDecision(json) {
  if (!json || typeof json !== "object") throw new Error("decide output is not an object");
  if (typeof json.retrieve !== "boolean") throw new Error("retrieve must be a boolean");
  const retrieve = json.retrieve;
  if (json.query !== null && typeof json.query !== "string") throw new Error("query must be string or null");
  if (typeof json.reason !== "string" || json.reason.length > 512) throw new Error("invalid reason");
  let query = json.query === null ? null : json.query.trim();
  if (query && query.length > 512) throw new Error("query exceeds 512 characters");
  if (!retrieve) query = null;
  if (retrieve && !query) throw new Error("retrieve=true but query is null");
  return {
    retrieve,
    query,
    reason: json.reason == null ? "" : String(json.reason),
  };
}

/** hop 模式：单次请求判定是否检索 + 给 query。坏 JSON 由 completeJson 重试一次。 */
export async function decideHop({ conversationId, recentTurns, pendingUserText }) {
  const json = await completeJson({
    model: decideModel(),
    systemPrompt: DECIDE_SYSTEM_PROMPT,
    userText: buildDecideUserPrompt({ recentTurns, pendingUserText }),
    timeoutMs: decideTimeoutMs(),
    maxTokens: 1024,
    schema: decisionSchema,
    validate: normalizeDecision,
  });
  return normalizeDecision(json);
}

/** board 记忆检索工具定义（agent 模式使用）。 */
function makeMemoryRetrieveTool(conversationId) {
  const baseUrl = process.env.BOARD_MEMORY_URL || DEFAULT_BOARD_MEMORY_URL;
  return {
    name: "memory_retrieve",
    label: "Retrieve long-term memory",
    description:
      "从长期记忆中检索与 query 相关的记忆片段。入参 query 应使用与用户相同的语言；返回 JSON 形式的记忆列表。",
    parameters: Type.Object({
      query: Type.String({ description: "检索词（与用户语言一致）" }),
    }),
    execute: async (_toolCallId, params) => {
      const url = `${baseUrl.replace(/\/+$/, "")}/api/memory/retrieve`;
      const res = await fetch(url, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ conversation_id: conversationId, query: params.query, top_k: 4 }),
        signal: requestSignal(10000),
      });
      const text = await res.text();
      if (!res.ok) throw new Error(`memory retrieve failed: HTTP ${res.status} ${text.slice(0, 200)}`);
      return { content: [{ type: "text", text }], details: { url, conversationId } };
    },
  };
}

/** agent 模式：Agent + tools 的完整 agent loop，最终仍要求严格 JSON 结论。 */
export async function decideAgent({ conversationId, recentTurns, pendingUserText }) {
  const guard = budgetHook(getApiKey(), 1024);
  const agent = new Agent({
    streamFn: (model, context, options) => streamSimple(model, context, { ...options, maxRetries: 0 }),
    initialState: {
      systemPrompt: DECIDE_AGENT_SYSTEM_PROMPT,
      model: decideModel(),
      thinkingLevel: "low",
    },
    getApiKey,
    onPayload: guard,
  });
  // 0.74.2 未提供独立的 registerTool API，工具通过 state.tools 注册。
  agent.state.tools = [makeMemoryRetrieveTool(conversationId)];

  const timeoutMs = Number(process.env.PI_AGENT_TIMEOUT_MS || 45000);
  const timer = setTimeout(() => agent.abort(), timeoutMs);
  const signal = requestSignal(timeoutMs);
  const abort = () => agent.abort();
  signal.addEventListener("abort", abort, { once: true });
  try {
    const userText = buildDecideAgentUserPrompt({ recentTurns, pendingUserText });
    let lastError = null;
    for (let attempt = 0; attempt < 2; attempt++) {
      const text =
        attempt === 0
          ? userText
          : `你的上轮回复不是合法 JSON（${String(lastError).slice(0, 120)}）。请只输出最终 JSON 对象，格式：{"retrieve": bool, "query": string|null, "reason": string}。`;
      await agent.prompt(text);
      if (guard.error) throw guard.error;
      const assistant = [...agent.state.messages].reverse().find((m) => m.role === "assistant");
      if (!assistant) throw new Error("agent produced no assistant message");
      if (assistant.stopReason === "error" || assistant.stopReason === "aborted") {
        throw new Error(`LLM call failed: ${assistant.errorMessage || assistant.stopReason}`);
      }
      try {
        return normalizeDecision(parseJsonStrict(messageText(assistant)));
      } catch (err) {
        lastError = err.message; // 坏 JSON：追加一条 nudge 再试一次
      }
    }
    throw new Error(`agent output is not valid JSON: ${lastError}`);
  } finally {
    clearTimeout(timer);
    signal.removeEventListener("abort", abort);
  }
}

/** /decide 入口：按 PI_AGENT_MODE 分流。 */
export async function decide(input) {
  const mode = (process.env.PI_AGENT_MODE || "hop").toLowerCase();
  if (mode === "agent") return decideAgent(input);
  if (mode === "hop") return decideHop(input);
  throw new Error(`unknown PI_AGENT_MODE: ${mode}`);
}
