import assert from "node:assert/strict";
import http from "node:http";
import { after, before, beforeEach, test } from "node:test";
import { assertBudget, BudgetError } from "./budget.mjs";
import { buildModel, completeJson } from "./aigw.mjs";
import { compactJournal, journalBlocks, normalizeCompact } from "./compact.mjs";
import { decideAgent, decideHop, normalizeDecision } from "./decide.mjs";

let server, base, model;
let generated = [], tokenized = [], output, limit, retryOverflow;
before(async () => {
  server = http.createServer(async (req, res) => {
    const parts = [];
    for await (const part of req) parts.push(part);
    const payload = JSON.parse(Buffer.concat(parts).toString() || "{}");
    res.setHeader("content-type", "application/json");
    if (req.url === "/get_server_info") return res.end(JSON.stringify({ context_length: limit }));
    if (req.url === "/v1/tokenize") {
      if (payload.stream) { res.statusCode = 501; return res.end("{}"); }
      tokenized.push(payload);
      const text = payload.prompt ?? JSON.stringify(payload.messages) + JSON.stringify(payload.tools || []);
      const count = retryOverflow && text.includes("上次回复") ? 20000 : text.length;
      return res.end(JSON.stringify({ count, max_model_len: 262144 }));
    }
    if (req.url === "/v1/chat/completions") {
      generated.push(payload);
      const answer = typeof output === "function" ? output(generated.length) : output;
      const content = typeof answer === "string" ? answer : JSON.stringify(answer);
      res.setHeader("content-type", "text/event-stream");
      const chunk = { id: "test", model: "test", object: "chat.completion.chunk", created: 0,
        choices: [{ index: 0, delta: { role: "assistant", content }, finish_reason: null }] };
      res.end(`data: ${JSON.stringify(chunk)}\n\ndata: ${JSON.stringify({ ...chunk,
        choices: [{ index: 0, delta: {}, finish_reason: "stop" }] })}\n\ndata: [DONE]\n\n`);
      return;
    }
    res.statusCode = 404; res.end("{}");
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  base = `http://127.0.0.1:${server.address().port}/v1`;
  Object.assign(process.env, { AIGW_API_KEY: "test-only", AIGW_COMPACT_BASE_URL: base,
    AIGW_DECIDE_BASE_URL: base, AIGW_LOCAL_NO_REASONING: "1", PI_CONTEXT_TOKENS: "16384" });
  model = buildModel("test", base);
});
after(async () => { server.closeAllConnections(); await new Promise((resolve) => server.close(resolve)); });
beforeEach(() => {
  generated = []; tokenized = []; limit = 16384; retryOverflow = false;
  output = { summary: "Conversation summarized.", pins: [] };
});

test("strict boolean and bounded output schemas", async () => {
  for (const retrieve of ["false", "true", 0, 1, null]) {
    assert.throws(() => normalizeDecision({ retrieve, query: null, reason: "test" }));
  }
  assert.equal(normalizeDecision({ retrieve: false, query: null, reason: "test" }).retrieve, false);
  for (const result of [{ summary: {}, pins: [] }, { summary: "x", pins: [7] },
    { summary: "x".repeat(201), pins: [] }, { summary: "x", pins: Array(17).fill("p") }]) {
    await assert.rejects(normalizeCompact(result, model));
  }
  const bounded = await normalizeCompact({ summary: "x", pins: ["p".repeat(200), "q".repeat(200)] }, model);
  assert.deepEqual(bounded.pins, ["p".repeat(200)]);
});

test("actual server limit overrides application and tokenizer metadata", async () => {
  limit = 2048;
  await assert.rejects(assertBudget(model, "test", {
    messages: [{ role: "user", content: "x".repeat(1000) }], max_tokens: 1024,
  }), BudgetError);
  assert.equal(generated.length, 0);
});

test("hop includes recent turns and checks the exact generated payload", async () => {
  output = { retrieve: true, query: "earlier camera", reason: "history reference" };
  assert.equal((await decideHop({ recentTurns: "camera model=FM2", pendingUserText: "which model?" })).retrieve, true);
  assert.equal(generated.length, 1);
  const checked = tokenized.find((p) => p.messages);
  assert.deepEqual(checked.messages, generated[0].messages);
  assert.ok(JSON.stringify(checked.messages).includes("camera model=FM2"));
});

test("oversized recent turns/current input never reach generation", async () => {
  for (const input of [{ recentTurns: "x".repeat(20000), pendingUserText: "hello" },
    { recentTurns: "hello", pendingUserText: "x".repeat(20000) }]) {
    await assert.rejects(decideHop(input), BudgetError);
  }
  assert.equal(generated.length, 0);
});

test("agent tool schema is included in final-payload budget", async () => {
  output = { retrieve: false, query: null, reason: "current view" };
  await decideAgent({ conversationId: "test", recentTurns: "hi", pendingUserText: "describe now" });
  assert.equal(generated.length, 1);
  assert.ok(tokenized[0].tools.length);
  assert.deepEqual(tokenized[0].tools, generated[0].tools);
  assert.equal(generated[0].max_tokens, 1024);
});

test("retry nudge is budgeted again", async () => {
  output = "not json"; retryOverflow = true;
  await assert.rejects(completeJson({ model, systemPrompt: "json", userText: "hello",
    timeoutMs: 1000, maxTokens: 1024 }), BudgetError);
  assert.equal(generated.length, 1);
});

test("compaction covers every QA including latest correction", async () => {
  const blocks = Array.from({ length: 8 }, (_, i) => `用户: QA-${i} ${"x".repeat(1500)}\n助手: reply-${i}`);
  blocks.push("用户: latest correction FM2 costs 2000 yuan\n助手: confirmed [interrupted]");
  const journal = blocks.join("\n");
  assert.equal(journalBlocks(journal).length, blocks.length);
  await compactJournal({ journal });
  assert.ok(generated.length > 1);
  for (const block of blocks) {
    assert.ok(generated.some((payload) => payload.messages.some((m) => m.content.includes(block))));
  }
  assert.ok(generated.at(-1).messages.at(-1).content.includes("latest correction"));
});

test("one oversized QA is rejected whole, not silently truncated", async () => {
  await assert.rejects(compactJournal({ journal: `用户: ${"x".repeat(20000)}\n助手: yes` }), BudgetError);
  assert.equal(generated.length, 0);
});

test("pins must be verbatim source excerpts", async () => {
  output = { summary: "hello", pins: ["fabricated identifier"] };
  await assert.rejects(compactJournal({ journal: "用户: hello\n助手: hi" }), /verbatim/);
});

test("pins can be validated against raw earlier chunks, not just previous selected pins", async () => {
  output = (call) => ({ summary: "EXACT_OLD_FACT", pins: call === 1 ? [] : ["EXACT_OLD_FACT"] });
  const journal = `用户: EXACT_OLD_FACT ${"a".repeat(1800)}\n助手: ok\n用户: ${"b".repeat(1800)}\n助手: ok`;
  const result = await compactJournal({ journal });
  assert.ok(generated.length >= 2);
  assert.deepEqual(result.pins, ["EXACT_OLD_FACT"]);
});
