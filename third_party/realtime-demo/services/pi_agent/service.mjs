// service.mjs — pi_agent loopback HTTP 决策/压缩服务入口。
//
// 路由（均为 JSON）：
//   GET  /health  -> {ok, model, mode}
//   POST /decide  -> {retrieve, query, reason}   (PI_AGENT_MODE=hop|agent)
//   POST /compact -> {summary, pins}
// 错误统一为 502 + {error}。

import http from "node:http";
import { ensureNoProxy, getApiKey } from "./aigw.mjs";
import { decide } from "./decide.mjs";
import { selectEvidence } from "./select.mjs";
import { compactJournal } from "./compact.mjs";
import { readiness } from "./readiness.mjs";
import { withDeadline } from "./deadline.mjs";

// 必须在任何出站请求之前：保证 no_proxy 覆盖 AIGW 主机（走代理会 TLS 失败）。
ensureNoProxy();

// 启动时读一次 key，尽早暴露配置错误（不打印 key 本身）。
try {
  getApiKey();
} catch (err) {
  console.error(`[pi_agent] failed to load AIGW api key: ${err.message}`);
  process.exit(1);
}

const PORT = Number(process.env.PI_PORT || 38082);
const HOST = "127.0.0.1";
const MAX_BODY_BYTES = 1024 * 1024; // 1MB 足够覆盖 journal

function mode() {
  return (process.env.PI_AGENT_MODE || "hop").toLowerCase();
}
function decideModelName() {
  return process.env.AIGW_DECIDE_MODEL || process.env.AIGW_MODEL || "Qwen3-4B-Instruct-2507";
}
function compactModelName() {
  return process.env.AIGW_COMPACT_MODEL || decideModelName();
}

function sendJson(res, status, obj) {
  const body = JSON.stringify(obj);
  res.writeHead(status, { "content-type": "application/json; charset=utf-8" });
  res.end(body);
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let size = 0;
    req.on("data", (c) => {
      size += c.length;
      if (size > MAX_BODY_BYTES) {
        chunks.length = 0;
        const error = new Error("request body exceeds 1 MiB");
        error.status = 413;
        reject(error);
        return;
      }
      chunks.push(c);
    });
    req.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
    req.on("error", reject);
  });
}

async function parseJsonBody(req) {
  const raw = await readBody(req);
  try {
    const body = JSON.parse(raw || "{}");
    if (!body || typeof body !== "object" || Array.isArray(body)) throw new Error("not an object");
    return body;
  } catch {
    const err = new Error("invalid JSON body");
    err.status = 400;
    throw err;
  }
}

function textField(body, key) {
  const value = body[key] ?? "";
  if (typeof value !== "string") {
    const error = new Error(`${key} must be a string`);
    error.status = 400;
    throw error;
  }
  return value;
}

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, `http://${HOST}:${PORT}`);
  const disconnected = new AbortController();
  res.on("close", () => {
    if (!res.writableEnded) disconnected.abort(Object.assign(new Error("client disconnected"), { status: 499 }));
  });
  const run = (defaultMs, action) => {
    const requested = Number(req.headers["x-memory-timeout-ms"] || defaultMs);
    if (!Number.isFinite(requested) || requested <= 0) {
      throw Object.assign(new Error("invalid X-Memory-Timeout-Ms"), { status: 400 });
    }
    return withDeadline(Math.min(defaultMs, Math.max(1, requested)), action, disconnected.signal);
  };
  try {
    if (req.method === "GET" && ["/health", "/ready"].includes(url.pathname)) {
      const state = await readiness();
      sendJson(res, state.ok ? 200 : 503, { ...state,
        model: decideModelName(), compact_model: compactModelName(), mode: mode() });
      return;
    }
    if (req.method === "GET" && url.pathname === "/live") {
      sendJson(res, 200, {
        ok: true,
        model: decideModelName(),
        compact_model: compactModelName(),
        mode: mode(),
      });
      return;
    }

    if (req.method === "POST" && url.pathname === "/decide") {
      const body = await parseJsonBody(req);
      const t0 = Date.now();
      const timeout = Number(process.env[mode() === "agent" ? "PI_AGENT_TIMEOUT_MS" : "PI_DECIDE_TIMEOUT_MS"] || (mode() === "agent" ? 45000 : 20000));
      const result = await run(timeout, () => decide({
        conversationId: String(body.conversation_id ?? ""),
        recentTurns: textField(body, "recent_turns"),
        pendingUserText: textField(body, "pending_user_text"),
      }));
      console.log(`[pi_agent] /decide mode=${mode()} retrieve=${result.retrieve} ${Date.now() - t0}ms`);
      sendJson(res, 200, result);
      return;
    }

    if (req.method === "POST" && url.pathname === "/select") {
      const body = await parseJsonBody(req);
      const result = await run(Number(process.env.PI_DECIDE_TIMEOUT_MS || 20000), () =>
        selectEvidence({ query: textField(body, "query"), candidates: body.candidates,
                         recentTurns: textField(body, "recent_turns") }));
      console.log(`[pi_agent] /select selected=${result.ids.length}`);
      sendJson(res, 200, result);
      return;
    }

    if (req.method === "POST" && url.pathname === "/compact") {
      const body = await parseJsonBody(req);
      const t0 = Date.now();
      const result = await run(Number(process.env.PI_COMPACT_TIMEOUT_MS || 60000), () => compactJournal({
        conversationId: String(body.conversation_id ?? ""),
        journal: textField(body, "journal"),
        summaryMaxTokens: body.summary_max_tokens ?? 200,
      }));
      console.log(`[pi_agent] /compact summary=${result.summary.length}字 pins=${result.pins.length} ${Date.now() - t0}ms`);
      sendJson(res, 200, result);
      return;
    }

    sendJson(res, 404, { error: `not found: ${req.method} ${url.pathname}` });
  } catch (err) {
    if (res.destroyed) return;
    const status = err.status || 502;
    console.error(`[pi_agent] ${req.method} ${url.pathname} -> ${status}: ${err.message}`);
    sendJson(res, status, { error: String(err.message || err) });
  }
});

server.listen(PORT, HOST, () => {
  console.log(`[pi_agent] listening on http://${HOST}:${PORT} mode=${mode()} model=${decideModelName()}`);
  console.log(`[pi_agent] Demo endpoint: MEMORY_PI_URL=http://${HOST}:${PORT}`);
  console.log(`[pi_agent] health: curl --fail http://${HOST}:${PORT}/health`);
});
