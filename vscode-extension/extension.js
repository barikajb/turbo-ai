"use strict";

const path = require("path");
const vscode = require("vscode");
const lib = require("./lib");

const PROPOSED_SCHEME = "turbo-proposed";

/** @type {vscode.WebviewPanel | undefined} */
let panel;
/** Conversation log shown in the panel. */
const log = [];
/** The fix currently shown in a diff editor, if any. */
let pending;
/** Proposed file contents served to the diff editor. */
const proposedDocs = new Map();

function config() {
  const c = vscode.workspace.getConfiguration("turbo");
  return {
    serverUrl: c.get("serverUrl", "http://127.0.0.1:8765"),
    token: c.get("token", ""),
    projectPath: c.get("projectPath", ""),
    contextLines: c.get("contextLines", 40),
    timeoutMs: Math.max(10, c.get("requestTimeoutSeconds", 600)) * 1000,
  };
}

function sessionId() {
  return lib.sessionIdFor(vscode.workspace.name);
}

// ---- panel ------------------------------------------------------------------

function showPanel() {
  if (!panel) {
    panel = vscode.window.createWebviewPanel("turbo", "TURBO", {
      viewColumn: vscode.ViewColumn.Beside,
      preserveFocus: true,
    }, { enableScripts: false, retainContextWhenHidden: true });
    panel.onDidDispose(() => (panel = undefined));
  } else {
    panel.reveal(undefined, true);
  }
  render();
}

function addEntry(kind, title, text) {
  const entry = { kind, title, text };
  log.push(entry);
  if (log.length > 60) log.shift();
  showPanel();
  return entry;
}

function updateEntry(entry, kind, text) {
  entry.kind = kind;
  entry.text = text;
  render();
}

function render() {
  if (!panel) return;
  const items = log
    .map((e) => `<section class="${e.kind}"><h3>${lib.escapeHtml(e.title)}</h3>${
      e.kind === "thinking" ? `<p class="muted">${lib.escapeHtml(e.text)}</p>` : lib.renderMarkdown(e.text)
    }</section>`)
    .join("");
  panel.webview.html = `<!DOCTYPE html>
<html><head><meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline';">
<style>
  body { font-family: var(--vscode-font-family); font-size: var(--vscode-font-size); color: var(--vscode-foreground); padding: 0 12px 24px; line-height: 1.5; }
  section { border-bottom: 1px solid var(--vscode-panel-border); padding: 8px 0 12px; }
  h3 { font-size: 0.85em; text-transform: uppercase; letter-spacing: .05em; color: var(--vscode-descriptionForeground); margin: 8px 0 4px; }
  .user h3 { color: var(--vscode-textLink-foreground); }
  .error h3, .error .text { color: var(--vscode-errorForeground); }
  .text { white-space: pre-wrap; margin: 4px 0; }
  .muted { color: var(--vscode-descriptionForeground); font-style: italic; }
  pre { background: var(--vscode-textCodeBlock-background); padding: 8px 10px; overflow-x: auto; border-radius: 4px; }
  code { font-family: var(--vscode-editor-font-family); font-size: var(--vscode-editor-font-size); }
  :not(pre) > code { background: var(--vscode-textCodeBlock-background); padding: 0 3px; border-radius: 3px; }
  .add { color: var(--vscode-gitDecoration-addedResourceForeground, #3fb950); }
  .del { color: var(--vscode-gitDecoration-deletedResourceForeground, #f85149); }
  .hunk { color: var(--vscode-textLink-foreground); }
  .meta { color: var(--vscode-descriptionForeground); }
</style></head>
<body>${items || '<p class="muted">Ask TURBO something, or select code and run Explain / Fix.</p>'}</body></html>`;
}

// ---- server calls -----------------------------------------------------------

async function callOnce(pathName, body, title) {
  const cfg = config();
  return vscode.window.withProgress(
    { location: vscode.ProgressLocation.Notification, title, cancellable: true },
    (_progress, token) => {
      const controller = new AbortController();
      token.onCancellationRequested(() => controller.abort());
      return lib.callServer({
        baseUrl: cfg.serverUrl,
        path: pathName,
        body,
        token: cfg.token,
        timeoutMs: cfg.timeoutMs,
        signal: controller.signal,
      });
    }
  );
}

/**
 * Call the server. If a free cloud limit is reached, ask the user what to do
 * (switch model now / wait for the reset / cancel) and retry as chosen.
 */
async function call(pathName, body, title) {
  for (let attempt = 0; attempt < 5; attempt++) {
    try {
      return await callOnce(pathName, body, title);
    } catch (error) {
      if (!(error instanceof lib.TurboError) || !error.decision) throw error;
      const decision = error.decision;
      const items = lib.decisionItems(decision);
      if (items.length === 0 || items.every((i) => i.option.kind === "cancel")) {
        throw new lib.TurboError(`${decision.message} ${decision.advice || ""}`.trim());
      }
      const picked = await vscode.window.showQuickPick(items, {
        title: `TURBO: ${decision.message}`,
        placeHolder: decision.advice || "Continue with the request?",
        ignoreFocusOut: true,
      });
      if (!picked || picked.option.kind === "cancel") {
        throw new lib.TurboError("Cancelled.", { cancelled: true });
      }
      const cfg = config();
      await lib.callServer({
        baseUrl: cfg.serverUrl,
        path: "/choose",
        body: { decision_id: decision.id, option: picked.index },
        token: cfg.token,
        timeoutMs: 10000,
      });
      if (picked.option.kind === "wait" && picked.option.wait_seconds > 0) {
        await waitWithCountdown(picked.option.wait_seconds, picked.option.model);
      }
    }
  }
  throw new lib.TurboError("Still at the free limit after several tries. Try again later.");
}

function waitWithCountdown(seconds, model) {
  return vscode.window.withProgress(
    { location: vscode.ProgressLocation.Notification, title: `Waiting for ${model} to reset`, cancellable: true },
    (progress, token) =>
      new Promise((resolve, reject) => {
        const end = Date.now() + seconds * 1000;
        const timer = setInterval(() => {
          const left = Math.max(0, (end - Date.now()) / 1000);
          progress.report({ message: `${Math.ceil(left)}s left` });
          if (left <= 0) {
            clearInterval(timer);
            resolve();
          }
        }, 500);
        token.onCancellationRequested(() => {
          clearInterval(timer);
          reject(new lib.TurboError("Cancelled.", { cancelled: true }));
        });
      })
  );
}

async function reportError(error, entry) {
  const message = error instanceof lib.TurboError ? error.message : String(error && error.message || error);
  if (entry) updateEntry(entry, error.cancelled ? "thinking" : "error", message);
  if (error && error.cancelled) return;
  if (error && error.serverDown) {
    const choice = await vscode.window.showErrorMessage(message, "Start Server");
    if (choice === "Start Server") await startServer();
    return;
  }
  vscode.window.showErrorMessage(`TURBO: ${message}`);
}

// ---- helpers ----------------------------------------------------------------

function getSelectionInfo(editor) {
  const doc = editor.document;
  const sel = editor.selection;
  const span = lib.fullLineSpan(
    {
      isEmpty: sel.isEmpty,
      startLine: sel.start.line,
      endLine: sel.end.line,
      endChar: sel.end.character,
    },
    doc.lineCount
  );
  const range = new vscode.Range(span.startLine, 0, span.endLine, doc.lineAt(span.endLine).text.length);
  let context = null;
  if (!span.wholeDocument) {
    const lines = [];
    for (let i = 0; i < doc.lineCount; i++) lines.push(doc.lineAt(i).text);
    context = lib.contextAround(lines, span.startLine, span.endLine, config().contextLines);
  }
  return {
    doc,
    range,
    code: doc.getText(range),
    context,
    wholeDocument: span.wholeDocument,
    filename: vscode.workspace.asRelativePath(doc.uri, false),
    language: doc.languageId,
    label: span.wholeDocument
      ? path.basename(doc.fileName)
      : `${path.basename(doc.fileName)}:${span.startLine + 1}-${span.endLine + 1}`,
  };
}

// ---- commands ---------------------------------------------------------------

async function askAi() {
  const editor = vscode.window.activeTextEditor;
  const hasSelection = editor && !editor.selection.isEmpty;
  const question = await vscode.window.showInputBox({
    prompt: hasSelection ? "Ask TURBO about the selected code" : "Ask TURBO anything (follow-ups remember the conversation)",
    placeHolder: "e.g. Why does this throw a KeyError?",
    ignoreFocusOut: true,
  });
  if (!question || !question.trim()) return;

  const body = { question, session_id: sessionId() };
  let title = "You";
  if (hasSelection) {
    const info = getSelectionInfo(editor);
    body.context = info.code;
    body.language = info.language;
    title = `You (about ${info.label})`;
  }
  addEntry("user", title, question);
  const entry = addEntry("thinking", "TURBO", "Thinking...");
  try {
    const data = await call("/ask", body, "TURBO is thinking...");
    updateEntry(entry, "assistant", data.answer);
  } catch (error) {
    await reportError(error, entry);
  }
}

async function explainSelection() {
  const editor = vscode.window.activeTextEditor;
  if (!editor) return vscode.window.showWarningMessage("Open a file first.");
  const info = getSelectionInfo(editor);
  if (!info.code.trim()) return vscode.window.showWarningMessage("Nothing to explain.");

  addEntry("user", "Explain", `\`${info.label}\``);
  const entry = addEntry("thinking", "TURBO", "Reading the code...");
  try {
    const data = await call(
      "/explain",
      {
        code: info.code,
        filename: info.filename,
        language: info.language,
        context: info.context,
        session_id: sessionId(),
      },
      `TURBO is explaining ${info.label}...`
    );
    updateEntry(entry, "assistant", data.answer);
  } catch (error) {
    await reportError(error, entry);
  }
}

async function fixSelection() {
  const editor = vscode.window.activeTextEditor;
  if (!editor) return vscode.window.showWarningMessage("Open a file first.");
  const info = getSelectionInfo(editor);
  if (!info.code.trim()) return vscode.window.showWarningMessage("Nothing to fix.");

  const instruction = await vscode.window.showInputBox({
    prompt: `Fix ${info.label}${info.wholeDocument ? " (whole file - select code to narrow it)" : ""}`,
    placeHolder: "Optional: describe the problem (e.g. crashes on empty list). Enter to let TURBO find it.",
    ignoreFocusOut: true,
  });
  if (instruction === undefined) return; // Esc

  await rejectFix(); // only one pending fix at a time
  addEntry("user", "Fix", `\`${info.label}\`${instruction ? `\n${instruction}` : ""}`);
  const entry = addEntry("thinking", "TURBO", "Looking for the bug...");
  const versionAtRequest = info.doc.version;

  let data;
  try {
    data = await call(
      "/fix",
      {
        code: info.code,
        filename: info.filename,
        language: info.language,
        instruction: instruction || null,
        context: info.context,
        session_id: sessionId(),
        start_line: info.range.start.line + 1,
      },
      `TURBO is fixing ${info.label}...`
    );
  } catch (error) {
    return reportError(error, entry);
  }

  const warnings = (data.warnings || []).map((w) => `**Warning:** ${w}`).join("\n");
  const shown = [data.explanation, warnings, data.changed ? "```diff\n" + data.diff + "```" : ""]
    .filter(Boolean)
    .join("\n\n");
  updateEntry(entry, "assistant", shown);

  if (!data.changed) {
    vscode.window.showInformationMessage("TURBO didn't suggest any changes. See the TURBO panel.");
    return;
  }
  if (info.doc.version !== versionAtRequest) {
    vscode.window.showWarningMessage("The file changed while TURBO was working. Run Fix Selection again.");
    return;
  }

  const doc = info.doc;
  const fullText = doc.getText();
  const startOffset = doc.offsetAt(info.range.start);
  const endOffset = doc.offsetAt(info.range.end);
  const proposedText = fullText.slice(0, startOffset) + data.fixed_code + fullText.slice(endOffset);

  const proposedUri = vscode.Uri.from({
    scheme: PROPOSED_SCHEME,
    path: doc.uri.path,
    query: `id=${Date.now()}`,
  });
  proposedDocs.set(proposedUri.toString(), proposedText);

  pending = { doc, range: info.range, fixedCode: data.fixed_code, version: doc.version, proposedUri };
  await vscode.commands.executeCommand("setContext", "turbo.fixPending", true);
  await vscode.commands.executeCommand(
    "vscode.diff",
    doc.uri,
    proposedUri,
    `TURBO fix: ${path.basename(doc.fileName)} (current ↔ proposed)`,
    { preview: true }
  );

  const thisFix = pending;
  const choice = await vscode.window.showInformationMessage(
    "Apply TURBO's fix? (You can also use the ✓ / ✕ buttons in the diff view.)",
    "Accept",
    "Reject"
  );
  if (pending !== thisFix) return; // already handled via the diff buttons
  if (choice === "Accept") await acceptFix();
  else if (choice === "Reject") await rejectFix();
}

async function closeDiff(proposedUri) {
  const key = proposedUri.toString();
  for (const group of vscode.window.tabGroups.all) {
    for (const tab of group.tabs) {
      const input = tab.input;
      if (input instanceof vscode.TabInputTextDiff && input.modified.toString() === key) {
        await vscode.window.tabGroups.close(tab);
      }
    }
  }
  proposedDocs.delete(key);
}

async function clearPending() {
  const done = pending;
  pending = undefined;
  await vscode.commands.executeCommand("setContext", "turbo.fixPending", false);
  if (done) await closeDiff(done.proposedUri);
  return done;
}

async function acceptFix() {
  if (!pending) return;
  if (pending.doc.isClosed || pending.doc.version !== pending.version) {
    await clearPending();
    vscode.window.showWarningMessage("The file changed after TURBO made the fix, so it wasn't applied. Run Fix Selection again.");
    return;
  }
  const fix = await clearPending();
  const edit = new vscode.WorkspaceEdit();
  edit.replace(fix.doc.uri, fix.range, fix.fixedCode);
  const ok = await vscode.workspace.applyEdit(edit);
  if (!ok) {
    vscode.window.showErrorMessage("TURBO couldn't apply the fix.");
    return;
  }
  const editor = await vscode.window.showTextDocument(fix.doc, { preserveFocus: false });
  editor.selection = new vscode.Selection(fix.range.start, fix.range.start);
  editor.revealRange(fix.range, vscode.TextEditorRevealType.InCenterIfOutsideViewport);
  vscode.window.setStatusBarMessage("TURBO fix applied (Ctrl+Z to undo).", 5000);
}

async function rejectFix() {
  if (pending) await clearPending();
}

async function resetConversation() {
  try {
    await lib.callServer({
      baseUrl: config().serverUrl,
      path: "/reset",
      body: { session_id: sessionId() },
      token: config().token,
      timeoutMs: 10000,
    });
    log.length = 0;
    render();
    vscode.window.showInformationMessage("TURBO conversation cleared.");
  } catch (error) {
    await reportError(error);
  }
}

async function startServer() {
  const projectPath = config().projectPath;
  if (!projectPath) {
    // Installed with pip/pipx: `turbo` is on PATH. Works in any shell.
    const terminal = vscode.window.createTerminal({ name: "TURBO server" });
    terminal.show(true);
    terminal.sendText("turbo serve");
    vscode.window.setStatusBarMessage(
      "Starting TURBO server... (if 'turbo' isn't found, install it or set turbo.projectPath)",
      8000
    );
    return;
  }
  // Source checkout: run its venv directly.
  const isWindows = process.platform === "win32";
  const python = isWindows
    ? path.join(projectPath, ".venv", "Scripts", "python.exe")
    : path.join(projectPath, ".venv", "bin", "python");
  // Run Python directly as the terminal process so it works whatever the
  // default shell is (PowerShell, cmd, Git Bash).
  const terminal = vscode.window.createTerminal({
    name: "TURBO server",
    cwd: projectPath,
    shellPath: python,
    shellArgs: [path.join(projectPath, "turbo.py"), "serve"],
  });
  terminal.show(true);
  vscode.window.setStatusBarMessage("Starting TURBO server...", 5000);
}

async function checkServerQuietly(statusItem) {
  try {
    const health = await lib.callServer({
      baseUrl: config().serverUrl,
      path: "/health",
      token: config().token,
      timeoutMs: 3000,
    });
    statusItem.text = "$(sparkle) TURBO";
    statusItem.tooltip = `TURBO ready - chat: ${health.chat_model}, code: ${health.code_model}` +
      (health.ollama ? "" : "\nWarning: server can't reach Ollama") +
      (health.missing_models && health.missing_models.length ? `\nMissing models: ${health.missing_models.join(", ")}` : "");
  } catch {
    statusItem.text = "$(circle-slash) TURBO";
    statusItem.tooltip = "TURBO server not running - click to ask anyway, or run 'TURBO: Start Server'";
  }
}

function activate(context) {
  context.subscriptions.push(
    vscode.workspace.registerTextDocumentContentProvider(PROPOSED_SCHEME, {
      provideTextDocumentContent: (uri) => proposedDocs.get(uri.toString()) || "",
    }),
    vscode.commands.registerCommand("turbo.ask", askAi),
    vscode.commands.registerCommand("turbo.explainSelection", explainSelection),
    vscode.commands.registerCommand("turbo.fixSelection", fixSelection),
    vscode.commands.registerCommand("turbo.acceptFix", acceptFix),
    vscode.commands.registerCommand("turbo.rejectFix", rejectFix),
    vscode.commands.registerCommand("turbo.resetConversation", resetConversation),
    vscode.commands.registerCommand("turbo.startServer", startServer),
    vscode.commands.registerCommand("turbo.showPanel", showPanel)
  );

  const statusItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
  statusItem.command = "turbo.ask";
  statusItem.text = "$(sparkle) TURBO";
  statusItem.show();
  context.subscriptions.push(statusItem);
  checkServerQuietly(statusItem);
  const timer = setInterval(() => checkServerQuietly(statusItem), 30000);
  context.subscriptions.push({ dispose: () => clearInterval(timer) });
}

function deactivate() {}

module.exports = { activate, deactivate };
