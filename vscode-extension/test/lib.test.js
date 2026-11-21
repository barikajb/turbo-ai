"use strict";
// Run: ELECTRON_RUN_AS_NODE=1 "<VS Code>/Code.exe" --test vscode-extension/test
// (or `node --test vscode-extension/test` if Node is installed)

const test = require("node:test");
const assert = require("node:assert/strict");
const lib = require("../lib");

test("fullLineSpan: empty selection means whole document", () => {
  assert.deepEqual(lib.fullLineSpan({ isEmpty: true }, 10), { startLine: 0, endLine: 9, wholeDocument: true });
});

test("fullLineSpan: selection ending at column 0 excludes that line", () => {
  const span = lib.fullLineSpan({ isEmpty: false, startLine: 2, endLine: 5, endChar: 0 }, 10);
  assert.deepEqual(span, { startLine: 2, endLine: 4, wholeDocument: false });
});

test("fullLineSpan: partial line selection keeps the line", () => {
  const span = lib.fullLineSpan({ isEmpty: false, startLine: 3, endLine: 3, endChar: 0 }, 10);
  assert.equal(span.endLine, 3);
});

test("contextAround: marker between before and after", () => {
  const lines = ["a", "b", "c", "d", "e"];
  assert.equal(lib.contextAround(lines, 2, 2, 1), "b\n>>> selected code goes here <<<\nd");
  assert.equal(lib.contextAround(lines, 0, 4, 3), null);
  assert.equal(lib.contextAround(lines, 1, 1, 0), null);
});

test("renderMarkdown escapes HTML outside and inside code", () => {
  const html = lib.renderMarkdown("Use <b>`<x>`</b>\n```js\nif (a < b) {}\n```");
  assert.ok(!html.includes("<b>"));
  assert.ok(html.includes("&lt;b&gt;"));
  assert.ok(html.includes("<code>&lt;x&gt;</code>"));
  assert.ok(html.includes("<pre><code>if (a &lt; b) {}</code></pre>"));
});

test("renderMarkdown colors diff lines", () => {
  const html = lib.renderMarkdown("```diff\n--- a/x\n+++ b/x\n@@ -1 +1 @@\n-old\n+new\n```");
  assert.ok(html.includes('<span class="del">-old</span>'));
  assert.ok(html.includes('<span class="add">+new</span>'));
  assert.ok(html.includes('<span class="hunk">'));
  assert.ok(html.includes('<span class="meta">--- a/x</span>'));
});

test("sessionIdFor", () => {
  assert.equal(lib.sessionIdFor("proj"), "vscode:proj");
  assert.equal(lib.sessionIdFor(undefined), "vscode:default");
});

function fakeResponse(status, body) {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => {
      if (body === undefined) throw new Error("bad json");
      return body;
    },
  };
}

test("callServer sends JSON, token and returns data", async () => {
  let seen;
  const data = await lib.callServer({
    baseUrl: "http://127.0.0.1:1/",
    path: "/ask",
    body: { question: "q" },
    token: "t",
    timeoutMs: 1000,
    fetchImpl: async (url, opts) => {
      seen = { url, opts };
      return fakeResponse(200, { answer: "a" });
    },
  });
  assert.deepEqual(data, { answer: "a" });
  assert.equal(seen.url, "http://127.0.0.1:1/ask");
  assert.equal(seen.opts.method, "POST");
  assert.equal(seen.opts.headers["X-Turbo-Token"], "t");
  assert.equal(JSON.parse(seen.opts.body).question, "q");
});

test("callServer GET when no body", async () => {
  let method;
  await lib.callServer({
    baseUrl: "http://x", path: "/health", timeoutMs: 1000,
    fetchImpl: async (_u, opts) => { method = opts.method; return fakeResponse(200, {}); },
  });
  assert.equal(method, "GET");
});

test("callServer surfaces server error messages", async () => {
  await assert.rejects(
    lib.callServer({
      baseUrl: "http://x", path: "/fix", body: {}, timeoutMs: 1000,
      fetchImpl: async () => fakeResponse(503, { error: "Model 'm' is not installed." }),
    }),
    (err) => err instanceof lib.TurboError && err.status === 503 && /not installed/.test(err.message)
  );
});

test("callServer: connection refused => serverDown", async () => {
  await assert.rejects(
    lib.callServer({
      baseUrl: "http://x", path: "/ask", body: {}, timeoutMs: 1000,
      fetchImpl: async () => { throw new TypeError("fetch failed"); },
    }),
    (err) => err.serverDown === true && /Is it running/.test(err.message)
  );
});

test("callServer: timeout", async () => {
  await assert.rejects(
    lib.callServer({
      baseUrl: "http://x", path: "/ask", body: {}, timeoutMs: 20,
      fetchImpl: (_u, opts) => new Promise((_resolve, reject) => {
        opts.signal.addEventListener("abort", () => reject(new Error("aborted")));
      }),
    }),
    (err) => !err.serverDown && !err.cancelled && /didn't answer/.test(err.message)
  );
});

test("callServer: user cancel", async () => {
  const controller = new AbortController();
  const promise = lib.callServer({
    baseUrl: "http://x", path: "/ask", body: {}, timeoutMs: 5000, signal: controller.signal,
    fetchImpl: (_u, opts) => new Promise((_resolve, reject) => {
      opts.signal.addEventListener("abort", () => reject(new Error("aborted")));
    }),
  });
  controller.abort();
  await assert.rejects(promise, (err) => err.cancelled === true);
});

test("callServer: unreadable reply", async () => {
  await assert.rejects(
    lib.callServer({
      baseUrl: "http://x", path: "/ask", body: {}, timeoutMs: 1000,
      fetchImpl: async () => fakeResponse(200, undefined),
    }),
    /unreadable/
  );
});

test("callServer passes the free-limit decision through", async () => {
  const decision = { id: "d1", message: "m reached its free limit", options: [{ kind: "switch", label: "Continue now with x", estimate_seconds: 2.5 }] };
  await assert.rejects(
    lib.callServer({
      baseUrl: "http://x", path: "/ask", body: {}, timeoutMs: 1000,
      fetchImpl: async () => fakeResponse(429, { error: "limit", decision }),
    }),
    (err) => err.status === 429 && err.decision && err.decision.id === "d1"
  );
});

test("decisionItems shows estimates and keeps option indexes", () => {
  const items = lib.decisionItems({ options: [
    { kind: "switch", label: "Continue now with b", estimate_seconds: 2.4 },
    { kind: "wait", label: "Wait for a", estimate_seconds: 95 },
    { kind: "cancel", label: "Cancel" },
  ] });
  assert.equal(items[0].description, "~2s to finish");
  assert.equal(items[1].description, "~2 min to finish");
  assert.equal(items[2].description, "");
  assert.deepEqual(items.map((i) => i.index), [0, 1, 2]);
});

test("formatSeconds", () => {
  assert.equal(lib.formatSeconds(0.2), "<1s");
  assert.equal(lib.formatSeconds(38), "38s");
  assert.equal(lib.formatSeconds(7200), "2.0 h");
});
