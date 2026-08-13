// Lightweight tests for app.js's pure logic — no framework, no
// dependencies: Node's built-in test runner + assert module only.
//
// Run with:  node --test frontend/app.test.js
//
// Only the DOM-free "pure" layer of app.js is under test here (see
// app.js's own comments on that split) — everything that touches
// `window`/`document` is skipped when app.js is require()'d from
// Node, since `typeof window !== "undefined"` is false there.

const test = require("node:test");
const assert = require("node:assert/strict");

const {
  isValidHex,
  normalizeHex,
  clampTopN,
  buildRequestBody,
  formatScore,
  formatRecommendations,
  extractErrorMessage,
  MAX_TOP_N,
  MIN_TOP_N,
  DEFAULT_TOP_N,
} = require("./app.js");

// --- hex validation ---

test("isValidHex accepts 6-digit hex with or without #", () => {
  assert.equal(isValidHex("#b23a2f"), true);
  assert.equal(isValidHex("b23a2f"), true);
  assert.equal(isValidHex("#B23A2F"), true);
});

test("isValidHex rejects malformed input", () => {
  assert.equal(isValidHex("#fff"), false); // 3-digit shorthand not supported
  assert.equal(isValidHex("not-a-color"), false);
  assert.equal(isValidHex("#gggggg"), false);
  assert.equal(isValidHex(""), false);
  assert.equal(isValidHex(undefined), false);
  assert.equal(isValidHex(null), false);
});

test("normalizeHex produces a lowercase #rrggbb string", () => {
  assert.equal(normalizeHex("B23A2F"), "#b23a2f");
  assert.equal(normalizeHex("#B23A2F"), "#b23a2f");
  assert.equal(normalizeHex("  #b23a2f  "), "#b23a2f");
});

test("normalizeHex returns null for invalid input", () => {
  assert.equal(normalizeHex("not-a-color"), null);
  assert.equal(normalizeHex(""), null);
});

// --- top_n clamping ---

test("clampTopN keeps in-range values as-is", () => {
  assert.equal(clampTopN(5), 5);
  assert.equal(clampTopN("10"), 10);
});

test("clampTopN clamps out-of-range values", () => {
  assert.equal(clampTopN(0), MIN_TOP_N);
  assert.equal(clampTopN(-5), MIN_TOP_N);
  assert.equal(clampTopN(999), MAX_TOP_N);
});

test("clampTopN falls back to the default for garbage input", () => {
  assert.equal(clampTopN("not-a-number"), DEFAULT_TOP_N);
  assert.equal(clampTopN(undefined), DEFAULT_TOP_N);
  assert.equal(clampTopN(NaN), DEFAULT_TOP_N);
});

// --- API request construction ---

test("buildRequestBody matches the API's expected shape", () => {
  const body = buildRequestBody(["#b23a2f", "#2f6b8e"], 5);
  assert.deepEqual(body, { colors: ["#b23a2f", "#2f6b8e"], top_n: 5 });
});

test("buildRequestBody clamps top_n through the same rules as clampTopN", () => {
  const body = buildRequestBody(["#b23a2f"], 999);
  assert.equal(body.top_n, MAX_TOP_N);
});

// --- formatting recommendations for rendering ---

test("formatRecommendations formats scores to 3 decimal places", () => {
  const formatted = formatRecommendations([
    { hex: "#412215", score: 1.9660081 },
    { hex: "#5b6227", score: 0.5 },
  ]);
  assert.deepEqual(formatted, [
    { hex: "#412215", scoreText: "1.966" },
    { hex: "#5b6227", scoreText: "0.500" },
  ]);
});

test("formatRecommendations handles an empty list", () => {
  assert.deepEqual(formatRecommendations([]), []);
});

test("formatRecommendations is defensive against a missing/malformed array", () => {
  assert.deepEqual(formatRecommendations(undefined), []);
  assert.deepEqual(formatRecommendations(null), []);
});

test("formatScore always shows 3 decimals", () => {
  assert.equal(formatScore(1), "1.000");
  assert.equal(formatScore(0.1234), "0.123");
});

// --- error handling ---

test("extractErrorMessage surfaces a FastAPI/Pydantic validation message", () => {
  const body = { detail: [{ loc: ["body", "colors", 0], msg: "invalid hex color 'zzz'", type: "value_error" }] };
  const message = extractErrorMessage(422, body);
  assert.match(message, /invalid hex color/);
});

test("extractErrorMessage surfaces a plain string detail", () => {
  assert.equal(extractErrorMessage(400, { detail: "bad request" }), "bad request");
});

test("extractErrorMessage falls back to a friendly generic message for 422 with no usable detail", () => {
  const message = extractErrorMessage(422, null);
  assert.match(message, /valid/i);
});

test("extractErrorMessage gives a friendly message for server errors", () => {
  const message = extractErrorMessage(500, null);
  assert.match(message, /server/i);
});

test("extractErrorMessage gives a friendly message for network failure (status 0)", () => {
  const message = extractErrorMessage(0, null);
  assert.match(message, /reach the server/i);
});

test("extractErrorMessage never leaks raw JSON to the message", () => {
  const body = { detail: [{ msg: "invalid hex color 'zzz'" }] };
  const message = extractErrorMessage(422, body);
  assert.equal(message.includes("{"), false);
  assert.equal(message.includes("loc"), false);
});
