import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { test } from "node:test";

function probe(extra) {
  const env = { ...process.env };
  for (const key of Object.keys(env)) if (key.startsWith("AIGW_")) delete env[key];
  Object.assign(env, extra);
  return spawnSync(process.execPath, ["--input-type=module", "-e",
    "import {getApiKey} from './aigw.mjs'; console.log(getApiKey())"], { cwd: new URL(".", import.meta.url), env, encoding: "utf8" });
}
test("explicit local mode needs no credential file", () => {
  const result = probe({ AIGW_AUTH_MODE: "local" });
  assert.equal(result.status, 0, result.stderr);
  assert.equal(result.stdout.trim(), "local-no-auth");
});
test("remote endpoint cannot silently use local no-auth mode", () => {
  const result = probe({ AIGW_AUTH_MODE: "local", AIGW_DECIDE_BASE_URL: "https://example.com/v1" });
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /loopback/);
});
test("token mode requires explicitly supplied credentials", () => {
  const result = probe({});
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /AIGW_API_KEY/);
  assert.doesNotMatch(result.stderr, /ai_gateway_key\.txt/);
});
