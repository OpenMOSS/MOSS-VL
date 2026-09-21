// Count the final OpenAI payload with the serving model's chat template.
import { requestSignal } from "./deadline.mjs";
export class BudgetError extends Error {
  constructor(message) { super(message); this.status = 413; }
}

export function contextLimit() {
  const value = Number(process.env.PI_CONTEXT_TOKENS || 16384);
  if (!Number.isSafeInteger(value) || value < 2048) throw new Error("invalid PI_CONTEXT_TOKENS");
  return value;
}

async function request(model, apiKey, path, body) {
  const base = model.baseUrl.replace(/\/+$/, "");
  const url = path === "tokenize" ? `${base}/tokenize` : `${base.replace(/\/v1$/, "")}/${path}`;
  const response = await fetch(url, {
    method: body ? "POST" : "GET",
    headers: { "content-type": "application/json", authorization: `Bearer ${apiKey}` },
    ...(body ? { body: JSON.stringify(body) } : {}),
    signal: requestSignal(5000),
  });
  if (!response.ok) throw new Error(`token budget unavailable: ${path} HTTP ${response.status}`);
  return response.json();
}

export async function countTokens(model, apiKey, payload) {
  // Streaming belongs to generation, not tokenization. SGLang dispatches
  // stream=true to an unimplemented tokenize stream handler (HTTP 501).
  const { stream, stream_options, ...textPayload } = payload;
  const result = await request(model, apiKey, "tokenize", { ...textPayload, model: model.id });
  if (!Number.isSafeInteger(result.count) || result.count < 0) throw new Error("invalid tokenizer count");
  return result.count;
}

export async function assertBudget(model, apiKey, payload) {
  const info = await request(model, apiKey, "get_server_info");
  const actualLimits = [info.context_length, info.max_req_input_len]
    .filter((n) => Number.isSafeInteger(n) && n > 0);
  if (!actualLimits.length) throw new Error("server does not expose an actual context limit");
  const limit = Math.min(contextLimit(), ...actualLimits);
  const output = payload.max_tokens ?? payload.max_completion_tokens;
  if (!Number.isSafeInteger(output) || output <= 0) throw new Error("missing output token reserve");
  const input = await countTokens(model, apiKey, payload);
  if (input + output + 256 > limit) {
    throw new BudgetError(`token budget exceeded: input=${input}, output=${output}, margin=256, context=${limit}`);
  }
  return input;
}

export function budgetHook(apiKey, maxTokens) {
  const hook = async (payload, model) => {
    hook.error = null;
    try {
      if (maxTokens) payload.max_tokens = maxTokens;
      await assertBudget(model, apiKey, payload);
      return payload;
    } catch (err) {
      hook.error = err;
      throw err;
    }
  };
  return hook;
}
