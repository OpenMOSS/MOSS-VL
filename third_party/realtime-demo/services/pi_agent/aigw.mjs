// aigw.mjs — AI Gateway (OpenAI chat-completions compatible) client helpers.
//
// - Reads the API key from a key file at startup (never logged).
// - Wraps @earendil-works/pi-ai (0.74.2) `completeSimple`, which for
//   `openai-completions` models maps the pi-ai `reasoning` option to the
//   native `reasoning_effort` chat-completions field (thinkingFormat: "openai"
//   is the default; verified against the gateway payload).
// - Falls back to native fetch only for one-off raw calls if ever needed.

import fs from "node:fs";
import { completeSimple } from "@earendil-works/pi-ai";
import { budgetHook, contextLimit } from "./budget.mjs";
import { requestSignal } from "./deadline.mjs";

export const DEFAULT_AIGW_BASE_URL = "http://127.0.0.1:38090/v1";

/**
 * Make sure no_proxy covers the AIGW host. Routing aigw.sotatts.online through
 * the corporate HTTP(S) proxy fails TLS, so append the host if missing.
 * Must run before any outbound request is made.
 */
export function ensureNoProxy(baseUrl = process.env.AIGW_BASE_URL || DEFAULT_AIGW_BASE_URL) {
  let host = null;
  try {
    host = new URL(baseUrl).hostname;
  } catch {
    return;
  }
  for (const name of ["no_proxy", "NO_PROXY"]) {
    const entries = (process.env[name] || "")
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean);
    if (!entries.some((e) => e === host || (e.startsWith(".") && host.endsWith(e)))) {
      entries.push(host);
    }
    process.env[name] = entries.join(",");
  }
}

let cachedApiKey = null;

/** @returns {string} API key read from AIGW_API_KEY or the key file. Never logged. */
export function getApiKey() {
  if (cachedApiKey) return cachedApiKey;
  if (process.env.AIGW_API_KEY) {
    cachedApiKey = process.env.AIGW_API_KEY.trim();
    if (cachedApiKey) return cachedApiKey;
  }
  if (process.env.AIGW_KEY_FILE) {
    cachedApiKey = fs.readFileSync(process.env.AIGW_KEY_FILE, "utf8").trim();
    if (!cachedApiKey) throw new Error("AIGW api key file is empty");
    return cachedApiKey;
  }
  if (process.env.AIGW_AUTH_MODE === "local") {
    for (const url of [process.env.AIGW_DECIDE_BASE_URL, process.env.AIGW_COMPACT_BASE_URL]) {
      const host = new URL(url || process.env.AIGW_BASE_URL || DEFAULT_AIGW_BASE_URL).hostname;
      if (!["127.0.0.1", "localhost", "[::1]"].includes(host)) throw new Error("local auth mode requires loopback endpoints");
    }
    return "local-no-auth";
  }
  throw new Error("Set AIGW_API_KEY/AIGW_KEY_FILE, or explicitly use AIGW_AUTH_MODE=local for loopback services");
}

/**
 * Build a pi-ai Model for a chat model served by the gateway.
 * compat overrides: the gateway does not understand `store`, and wants
 * `max_tokens` (not `max_completion_tokens`).
 * @param {string} modelId
 * @param {string} [baseUrlOverride] — per-role base URL (e.g. decide on a local
 *   small instruct model while compact stays on the shared gateway).
 */
export function buildModel(modelId, baseUrlOverride) {
  return {
    id: modelId,
    name: `${modelId} (via OpenAI-compatible endpoint)`,
    api: "openai-completions",
    provider: "aigw",
    baseUrl: baseUrlOverride || process.env.AIGW_BASE_URL || DEFAULT_AIGW_BASE_URL,
    reasoning: true,
    input: ["text"],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: contextLimit(),
    maxTokens: 8192,
    compat: {
      supportsStore: false,
      maxTokensField: "max_tokens",
      // 非 OpenAI 官方端点不识别 developer role；老老实实下 system。
      supportsDeveloperRole: false,
    },
  };
}

/** Extract concatenated text blocks from an AssistantMessage. */
export function messageText(message) {
  return (message.content || [])
    .filter((c) => c.type === "text")
    .map((c) => c.text)
    .join("")
    .trim();
}

/**
 * Run one non-tool completion via pi-ai and return plain assistant text.
 * Throws on LLM errors / timeout (caller maps to 502).
 */
export async function completeText({ model, systemPrompt, userText, timeoutMs, maxTokens, schema }) {
  const guard = budgetHook(getApiKey());
  const message = await completeSimple(
    model,
    {
      systemPrompt,
      messages: [{ role: "user", content: userText, timestamp: Date.now() }],
    },
    {
      apiKey: getApiKey(),
      reasoning: process.env.AIGW_LOCAL_NO_REASONING === "1" ? undefined : "low", // -> reasoning_effort: "low"
      temperature: 0,
      timeoutMs,
      signal: requestSignal(timeoutMs),
      maxRetries: 0,
      maxTokens: maxTokens ?? 2048,
      onPayload: async (payload, servedModel) => {
        if (schema) payload.response_format = {
          type: "json_schema", json_schema: { name: "memory_result", strict: true, schema },
        };
        return guard(payload, servedModel);
      },
    },
  );
  if (guard.error) throw guard.error;
  if (message.stopReason === "error" || message.stopReason === "aborted") {
    throw new Error(`LLM call failed: ${message.errorMessage || message.stopReason}`);
  }
  const text = messageText(message);
  if (process.env.PI_LOG_LLM === "1") {
    console.log(`[pi_agent][llm-raw] stop=${message.stopReason} text=${JSON.stringify(text.slice(0, 400))}`);
  }
  return text;
}

/** Strip markdown fences / surrounding prose and JSON.parse. Throws on bad JSON. */
export function parseJsonStrict(text) {
  let s = String(text || "").trim();
  const fence = s.match(/```(?:json)?\s*([\s\S]*?)```/i);
  if (fence) s = fence[1].trim();
  const start = s.indexOf("{");
  const end = s.lastIndexOf("}");
  if (start === -1 || end === -1 || end <= start) {
    throw new Error("response contains no JSON object");
  }
  return JSON.parse(s.slice(start, end + 1));
}

/**
 * completeText + strict JSON, with exactly one retry on bad JSON output
 * (the retry re-asks with an explicit "you returned invalid JSON" nudge).
 */
export async function completeJson(params) {
  const deadline = Date.now() + params.timeoutMs;
  let userText = params.userText;
  for (let attempt = 0; attempt < 2; attempt++) {
    const timeoutMs = deadline - Date.now();
    if (timeoutMs <= 0) throw new Error("JSON completion deadline exceeded");
    const text = await completeText({ ...params, userText, timeoutMs });
    try {
      const json = parseJsonStrict(text);
      return params.validate ? await params.validate(json) : json;
    } catch (err) {
      if (attempt === 1) throw err;
      userText = `${params.userText}\n\n` +
        `# 上次回复未通过 JSON 校验（${String(err.message).slice(0, 120)}）。\n` +
        "请重新回答，严格遵守字段类型与长度上限，只输出一个 JSON 对象，不要带解释或代码块。";
    }
  }
}
