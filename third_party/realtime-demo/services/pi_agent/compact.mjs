import { buildModel, completeJson, getApiKey } from "./aigw.mjs";
import { assertBudget, BudgetError, countTokens } from "./budget.mjs";
import { COMPACT_SYSTEM_PROMPT, buildCompactUserPrompt } from "./prompts.mjs";
import { compactSchema } from "./schemas.mjs";

export async function normalizeCompact(json, model, summaryMaxTokens = 200) {
  if (!json || typeof json.summary !== "string" || !json.summary.trim()) {
    throw new Error("compact summary must be a non-empty string");
  }
  if (!Array.isArray(json.pins) || json.pins.length > 16 ||
      json.pins.some((p) => typeof p !== "string" || !p.trim() || p.length > 256)) {
    throw new Error("compact pins must contain at most 16 non-empty strings of at most 256 characters");
  }
  const summary = json.summary.trim();
  const candidates = [...new Set(json.pins)];
  if (summary.length > 200) throw new Error("compact summary exceeds 200 characters");
  const count = (text) => countTokens(model, getApiKey(), { prompt: text, add_special_tokens: false });
  if (await count(summary) > summaryMaxTokens) throw new Error("compact summary exceeds token budget");
  const pins = [];
  for (const pin of candidates) {
    if (await count([...pins, pin].join("\n")) <= 320) pins.push(pin);
  }
  if (pins.length < candidates.length) {
    console.warn(`[pi_agent] compact pins bounded: kept=${pins.length}, proposed=${candidates.length}`);
  }
  return { summary, pins };
}

export function journalBlocks(journal) {
  // Python emits one line per utterance. Keep a user turn and all following
  // assistant replies indivisible, including interrupted conversation.
  return journal.split(/\n(?=(?:用户|user):)/).filter((block) => block.trim());
}

export async function compactJournal({ journal, summaryMaxTokens = 200 }) {
  if (typeof journal !== "string" || !journal.trim()) throw new Error("journal must be non-empty text");
  if (!Number.isSafeInteger(summaryMaxTokens) || summaryMaxTokens < 16 || summaryMaxTokens > 200) {
    throw new Error("summary_max_tokens must be an integer between 16 and 200");
  }
  const model = buildModel(process.env.AIGW_COMPACT_MODEL || process.env.AIGW_MODEL || "Qwen3-4B-Instruct-2507",
    process.env.AIGW_COMPACT_BASE_URL || undefined);
  const systemPrompt = `${COMPACT_SYSTEM_PROMPT}\n` +
    `summary <= ${summaryMaxTokens} tokens and <= 200 characters. ` +
    "pins: at most 16 strings, each <= 256 characters, combined <= 320 tokens. " +
    "Order pins by importance: corrections and pending commitments first; newer corrections supersede older facts. " +
    "When previous compact state is provided, merge it with ALL new conversation blocks.";
  const blocks = journalBlocks(journal);
  const chunkTokens = Number(process.env.PI_COMPACT_CHUNK_TOKENS || 4096);
  if (!Number.isSafeInteger(chunkTokens) || chunkTokens < 1024) throw new Error("invalid PI_COMPACT_CHUNK_TOKENS");
  const timeoutMs = Number(process.env.PI_COMPACT_TIMEOUT_MS || 60000);
  const deadline = Date.now() + timeoutMs;
  let state = null;
  let offset = 0;
  for (let chunk = 0; offset < blocks.length; chunk++) {
    if (Date.now() >= deadline) throw new Error("compact job deadline exceeded; no partial result returned");
    if (chunk >= 32) throw new BudgetError("journal needs more than 32 chunks; no partial result returned");
    const prompt = (end) => buildCompactUserPrompt(
      `${state ? `Previous compact state:\n${JSON.stringify(state)}\nNew conversation:\n` : ""}` +
      blocks.slice(offset, end).join("\n"));
    let low = offset + 1, high = blocks.length, end = offset;
    while (low <= high) {
      if (Date.now() >= deadline) throw new Error("compact job deadline exceeded; no partial result returned");
      const mid = Math.floor((low + high) / 2);
      try {
        // Leave an additional 256 tokens for a possible JSON-repair nudge.
        const inputTokens = await assertBudget(model, getApiKey(), { model: model.id, max_tokens: 4352,
          messages: [{ role: "system", content: systemPrompt }, { role: "user", content: prompt(mid) }] });
        if (inputTokens > chunkTokens) throw new BudgetError("compact chunk exceeds token budget");
        end = mid; low = mid + 1;
      } catch (err) {
        if (!(err instanceof BudgetError)) throw err;
        high = mid - 1;
      }
    }
    if (end === offset) throw new BudgetError("one QA block exceeds input budget; no partial result returned");
    // A fact may survive in the rolling summary without being selected as a
    // pin in the previous chunk. Validate against raw history already covered.
    const source = blocks.slice(0, end).join("\n");
    const next = await completeJson({ model, systemPrompt, userText: prompt(end), schema: compactSchema,
      timeoutMs: Math.max(1, deadline - Date.now()), maxTokens: 4096,
      validate: async (json) => {
        const result = await normalizeCompact(json, model, summaryMaxTokens);
        if (result.pins.some((pin) => !source.includes(pin))) throw new Error("compact pins must be verbatim journal excerpts");
        return result;
      },
    });
    state = next;
    offset = end;
  }
  return state;
}
