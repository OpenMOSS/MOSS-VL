export const compactSchema = {
  type: "object", additionalProperties: false, required: ["summary", "pins"],
  properties: {
    summary: { type: "string", minLength: 1, maxLength: 200 },
    pins: { type: "array", maxItems: 16, items: { type: "string", minLength: 1, maxLength: 256 } },
  },
};

export const decisionSchema = {
  type: "object", additionalProperties: false, required: ["retrieve", "query", "reason"],
  properties: {
    retrieve: { type: "boolean" },
    query: { anyOf: [{ type: "string", maxLength: 512 }, { type: "null" }] },
    reason: { type: "string", maxLength: 512 },
  },
};
