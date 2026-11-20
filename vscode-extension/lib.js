// Pure helpers (no `vscode` import) so they can be unit tested with plain Node.
"use strict";

class TurboError extends Error {
  constructor(message, { status = 0, serverDown = false, cancelled = false } = {}) {
    super(message);
    this.name = "TurboError";
    this.status = status;
    this.serverDown = serverDown;
    this.cancelled = cancelled;
  }
}

/**
 * POST/GET JSON to the TURBO server. Throws TurboError with a message that is
 * safe to show to the user.
 */
async function callServer({ baseUrl, path, body, token, timeoutMs, signal, fetchImpl }) {
  const doFetch = fetchImpl || fetch;
  const controller = new AbortController();
  let timedOut = false;
  const timer = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, timeoutMs);
  const onAbort = () => controller.abort();
  if (signal) signal.addEventListener("abort", onAbort);

  const headers = { "Content-Type": "application/json" };
  if (token) headers["X-Turbo-Token"] = token;

  let response;
  try {
    response = await doFetch(baseUrl.replace(/\/+$/, "") + path, {
      method: body === undefined ? "GET" : "POST",
      headers,
      body: body === undefined ? undefined : JSON.stringify(body),
      signal: controller.signal,
    });
  } catch (error) {
    if (timedOut) {
      throw new TurboError(
        `TURBO didn't answer within ${Math.round(timeoutMs / 1000)}s. Try a smaller selection or raise turbo.requestTimeoutSeconds.`
      );
    }
    if (controller.signal.aborted) {
      throw new TurboError("Cancelled.", { cancelled: true });
    }
    throw new TurboError(`Can't reach the TURBO server at ${baseUrl}. Is it running?`, { serverDown: true });
  } finally {
    clearTimeout(timer);
    if (signal) signal.removeEventListener("abort", onAbort);
  }

  let data;
  try {
    data = await response.json();
  } catch {
    throw new TurboError(`TURBO server sent an unreadable reply (HTTP ${response.status}).`, {
      status: response.status,
    });
  }
  if (!response.ok) {
    throw new TurboError(data && data.error ? data.error : `HTTP ${response.status}`, {
      status: response.status,
    });
  }
  return data;
}

/**
 * Expand a selection to whole lines (0-based, inclusive). An empty selection
 * means the whole document. A selection ending at column 0 of a later line
 * doesn't include that line.
 */
function fullLineSpan(sel, lineCount) {
  if (sel.isEmpty) {
    return { startLine: 0, endLine: Math.max(0, lineCount - 1), wholeDocument: true };
  }
  let endLine = sel.endLine;
  if (sel.endChar === 0 && endLine > sel.startLine) endLine -= 1;
  return { startLine: sel.startLine, endLine, wholeDocument: false };
}

/** Lines around the selection, with a marker where the selection sits. */
function contextAround(lines, startLine, endLine, count) {
  if (count <= 0) return null;
  const before = lines.slice(Math.max(0, startLine - count), startLine);
  const after = lines.slice(endLine + 1, endLine + 1 + count);
  if (before.length === 0 && after.length === 0) return null;
  return [...before, ">>> selected code goes here <<<", ...after].join("\n");
}

function escapeHtml(text) {
  return String(text)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function renderInline(text) {
  return escapeHtml(text)
    .replace(/`([^`\n]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>")
    .replace(/^#{1,6} (.+)$/gm, "<strong>$1</strong>");
}

function renderCode(code, lang) {
  if (lang === "diff") {
    const lines = code.split("\n").map((line) => {
      let cls = "";
      if (line.startsWith("+++") || line.startsWith("---")) cls = "meta";
      else if (line.startsWith("+")) cls = "add";
      else if (line.startsWith("-")) cls = "del";
      else if (line.startsWith("@@")) cls = "hunk";
      return cls ? `<span class="${cls}">${escapeHtml(line)}</span>` : escapeHtml(line);
    });
    return `<pre><code>${lines.join("\n")}</code></pre>`;
  }
  return `<pre><code>${escapeHtml(code)}</code></pre>`;
}

/** Very small, safe Markdown renderer: code fences, inline code, bold. */
function renderMarkdown(markdown) {
  const fenceRe = /^(`{3,})([^\n`]*)\n([\s\S]*?)^\1[ \t]*$/gm;
  let html = "";
  let last = 0;
  let match;
  while ((match = fenceRe.exec(markdown)) !== null) {
    html += `<div class="text">${renderInline(markdown.slice(last, match.index).trim())}</div>`;
    html += renderCode(match[3].replace(/\n$/, ""), match[2].trim().toLowerCase());
    last = match.index + match[0].length;
  }
  html += `<div class="text">${renderInline(markdown.slice(last).trim())}</div>`;
  return html.replace(/<div class="text"><\/div>/g, "");
}

function sessionIdFor(workspaceName) {
  return `vscode:${workspaceName || "default"}`;
}

module.exports = {
  TurboError,
  callServer,
  fullLineSpan,
  contextAround,
  escapeHtml,
  renderMarkdown,
  sessionIdFor,
};
