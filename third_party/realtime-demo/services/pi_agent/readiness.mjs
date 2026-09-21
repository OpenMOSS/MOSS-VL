import { createHash } from "node:crypto";
import { readdirSync, readFileSync } from "node:fs";
import { buildModel, getApiKey } from "./aigw.mjs";
import { assertBudget } from "./budget.mjs";
import { requestSignal, withDeadline } from "./deadline.mjs";

const directory = new URL("./", import.meta.url);
const hash = createHash("sha256");
for (const file of readdirSync(directory).filter((f) => f.endsWith(".mjs") && !f.endsWith(".test.mjs")).sort()) {
  hash.update(file); hash.update(readFileSync(new URL(file, directory)));
}
export const implementation = hash.digest("hex");

export function configuration() {
  const decide = buildModel(process.env.AIGW_DECIDE_MODEL || process.env.AIGW_MODEL || "Qwen3-4B-Instruct-2507",
    process.env.AIGW_DECIDE_BASE_URL);
  const compact = buildModel(process.env.AIGW_COMPACT_MODEL || process.env.AIGW_MODEL || "Qwen3-4B-Instruct-2507",
    process.env.AIGW_COMPACT_BASE_URL);
  return {
    PI_PORT: process.env.PI_PORT || "38080", AIGW_DECIDE_BASE_URL: decide.baseUrl,
    AIGW_COMPACT_BASE_URL: compact.baseUrl, AIGW_DECIDE_MODEL: decide.id,
    AIGW_COMPACT_MODEL: compact.id, AIGW_LOCAL_NO_REASONING: process.env.AIGW_LOCAL_NO_REASONING || "0",
    PI_CONTEXT_TOKENS: process.env.PI_CONTEXT_TOKENS || "16384",
    PI_COMPACT_CHUNK_TOKENS: process.env.PI_COMPACT_CHUNK_TOKENS || "4096",
    PI_COMPACT_TIMEOUT_MS: process.env.PI_COMPACT_TIMEOUT_MS || "60000",
    PI_DECIDE_TIMEOUT_MS: process.env.PI_DECIDE_TIMEOUT_MS || "20000",
    PI_AGENT_MODE: process.env.PI_AGENT_MODE || "hop",
    AIGW_AUTH_MODE: process.env.AIGW_AUTH_MODE || "token",
    PI_AGENT_TIMEOUT_MS: process.env.PI_AGENT_TIMEOUT_MS || "45000",
    BOARD_MEMORY_URL: process.env.BOARD_MEMORY_URL || "http://127.0.0.1:8081",
  };
}

export async function readiness() {
  const config = configuration();
  const roles = ["DECIDE", "COMPACT"];
  const checked = new Map();
  try {
    await withDeadline(5000, async () => {
      for (const role of roles) {
        const base = config[`AIGW_${role}_BASE_URL`].replace(/\/+$/, "");
        const id = config[`AIGW_${role}_MODEL`];
        const key = `${base}:${id}`;
        if (checked.has(key)) continue;
        const headers = { authorization: `Bearer ${getApiKey()}` };
        const health = await fetch(`${base.replace(/\/v1$/, "")}/health_generate`, { headers, signal: requestSignal() });
        if (!health.ok) throw new Error(`${role} model readiness HTTP ${health.status}`);
        await health.arrayBuffer();
        const response = await fetch(`${base}/models`, { headers, signal: requestSignal() });
        if (!response.ok) throw new Error(`${role} model list HTTP ${response.status}`);
        const models = await response.json();
        if (!models.data?.some((model) => model.id === id || model.id.split("/").at(-1) === id)) {
          throw new Error(`${role} configured model is not served`);
        }
        await assertBudget(buildModel(id, base), getApiKey(), {
          messages: [{ role: "user", content: "ready" }], max_tokens: 16,
        });
        checked.set(key, true);
      }
    });
    return { ok: true, configuration: config, implementation };
  } catch (error) {
    return { ok: false, error: error.message, configuration: config, implementation };
  }
}
