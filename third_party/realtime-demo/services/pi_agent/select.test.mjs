import assert from "node:assert/strict";
import test from "node:test";
import { normalizeSelection, validateCandidates } from "./select.mjs";

test("selection accepts only supplied integer ids", () => {
  const candidates = [{ id: 3, text: "fact" }];
  assert.deepEqual(normalizeSelection({ ids: [3, 3] }, candidates), { ids: [3] });
  assert.deepEqual(normalizeSelection({ ids: [] }, candidates), { ids: [] });
  for (const ids of [[4], ["3"], [true], null]) {
    assert.throws(() => normalizeSelection({ ids }, candidates));
  }
});

test("candidate payload is bounded", () => {
  assert.throws(() => validateCandidates(Array(17).fill({ id: 1, text: "a" })));
  assert.throws(() => validateCandidates([{ id: 1, text: "a".repeat(2001) }]));
  assert.throws(() => validateCandidates(null));
  assert.deepEqual(validateCandidates([]), []);
});
