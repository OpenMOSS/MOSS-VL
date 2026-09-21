import assert from "node:assert/strict";
import http from "node:http";
import { after, before, test } from "node:test";
import { readiness } from "./readiness.mjs";
import { withDeadline, requestSignal } from "./deadline.mjs";
import { completeJson, buildModel } from "./aigw.mjs";
import { decideAgent } from "./decide.mjs";

let server, base, health = 200, modelId = "Qwen3-4B-Instruct-2507", closed = 0, calls = 0;
before(async () => {
  server = http.createServer(async (req, res) => {
    for await (const part of req) { /* consume test payload */ }
    if (req.url === "/slow" || req.url === "/v1/chat/completions") {
      calls++;
      res.on("close", () => { closed++; });
      return;
    }
    let data = {};
    if (req.url === "/health_generate") res.statusCode = health;
    if (req.url === "/v1/models") data = { data: [{ id: `/models/${modelId}` }] };
    if (req.url === "/get_server_info") data = { context_length: 16384 };
    if (req.url === "/v1/tokenize") data = { count: 20 };
    res.setHeader("content-type", "application/json"); res.end(JSON.stringify(data));
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  base = `http://127.0.0.1:${server.address().port}`;
  Object.assign(process.env, { AIGW_API_KEY: "test-only", AIGW_DECIDE_BASE_URL: base + "/v1",
    AIGW_COMPACT_BASE_URL: base + "/v1", AIGW_DECIDE_MODEL: modelId, AIGW_COMPACT_MODEL: modelId });
});
after(async () => { server.closeAllConnections(); await new Promise((resolve) => server.close(resolve)); });

test("readiness checks model generation and identity", async () => {
  assert.equal((await readiness()).ok, true);
  health = 503;
  assert.equal((await readiness()).ok, false);
  health = 200; modelId = "wrong-model";
  assert.equal((await readiness()).ok, false);
  modelId = "Qwen3-4B-Instruct-2507";
});

test("deadline aborts the underlying network request", async () => {
  const start = Date.now();
  await assert.rejects(withDeadline(100, () => fetch(base + "/slow", { signal: requestSignal() })),
    (error) => error.status === 504);
  assert.ok(Date.now() - start < 500);
  await new Promise((resolve) => setTimeout(resolve, 50));
  assert.ok(closed > 0);
});

test("SDK timeout cannot be multiplied by automatic retries", async () => {
  calls = 0;
  const start = Date.now();
  await assert.rejects(withDeadline(150, () => completeJson({ model: buildModel(modelId, base + "/v1"),
    systemPrompt: "json", userText: "hi", timeoutMs: 150, maxTokens: 16 })));
  assert.ok(Date.now() - start < 600);
  assert.equal(calls, 1);
});

test("agent cancellation shares the HTTP deadline", async () => {
  calls = 0;
  const start = Date.now();
  await assert.rejects(withDeadline(150, () => decideAgent({ conversationId: "test", recentTurns: "", pendingUserText: "hi" })));
  assert.ok(Date.now() - start < 600);
  assert.equal(calls, 1);
});
