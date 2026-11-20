# TURBO - local coding assistant

TURBO answers questions, explains code, and fixes bugs (as unified diffs) using
models running on your own machine through [Ollama](https://ollama.com).
No API keys and no paid services. Your code never leaves your computer.

Use it from:
- **the terminal**: `turbo ask`, `turbo chat`, `turbo explain`, `turbo fix`
- **VS Code**: *Ask AI*, *Explain Selection*, *Fix Selection* (with a diff you accept or reject)

```
VS Code extension ──HTTP (127.0.0.1)──▶ turbo serve ─┐
terminal: turbo ask/fix ─────────────────────────────┼─▶ turbo_core ──▶ Ollama (local models)
                                                     └── SQLite conversation history (~/.turbo)
```

## Install

You need **Python 3.10+** and **[Ollama](https://ollama.com)** (free). It works on
Windows, macOS and Linux.

1. **Install Ollama** from https://ollama.com and start it. Then download the models
   (about 6.7 GB in total, one time only):
   ```
   ollama pull qwen2.5-coder:7b
   ollama pull llama3.2:3b
   ```
   `qwen2.5-coder:7b` handles explain/fix. `llama3.2:3b` is the faster model for plain
   questions. On a slower computer, use `qwen2.5-coder:3b` instead (see *Settings*).

2. **Install TURBO** with [pipx](https://pipx.pypa.io), which puts the `turbo` command
   on your PATH in its own isolated environment:
   ```
   pip install --user pipx
   python -m pipx ensurepath        # then open a new terminal
   pipx install git+https://github.com/barikajb/turbo-ai
   ```
   Without pipx: `pip install --user git+https://github.com/barikajb/turbo-ai`

3. **Check everything is ready**:
   ```
   turbo doctor
   ```
   It checks Ollama, the models and the server port, and tells you how to fix anything missing.

4. **Start chatting**: type `turbo`.

To update later: `pipx upgrade turbo-ai`. To remove: `pipx uninstall turbo-ai`.

### Settings

Everything has a default. To change something, create `~/.turbo/.env` (on Windows,
`C:\Users\<you>\.turbo\.env`) using [`.env.example`](.env.example) as a template, for example:
```
TURBO_CODE_MODEL=qwen2.5-coder:3b
```
Real environment variables override the file. `~/.turbo` also holds your conversation
history (`turbo.db`). Set `TURBO_HOME` to use a different folder.

### Developing TURBO (from a clone)

```
git clone https://github.com/barikajb/turbo-ai
cd turbo-ai
python -m venv .venv
.venv\Scripts\python -m pip install -e .        # macOS/Linux: .venv/bin/python
```
`pip install -e .` installs the `turbo` command from your clone, so edits take effect
immediately. On Windows you can instead add the clone to your PATH and use `turbo.cmd`.

## Terminal

```
turbo ask "What's the difference between a list and a tuple?"
turbo ask "What does this do?" -f app.py -l 10-30
turbo explain app.py -l 10-30
turbo explain app.py -q "Why is this slow?"
turbo fix app.py -l 10-30                       # explanation + colored diff, file untouched
turbo fix app.py -l 10-30 -m "crashes on empty list"
turbo fix app.py -l 10-30 --apply               # asks y/N, then writes
type app.py | turbo fix -                       # from stdin
turbo chat                                      # interactive; /file app.py 10-30, /paste, /reset, /exit
turbo reset                                     # forget the conversation
turbo doctor                                    # check Ollama, models, port
```

Follow-ups remember the conversation: run `turbo fix ...`, then
`turbo ask "why was that wrong?"`. Use `-s NAME` for separate conversations
(the default is `cli`). Diffs can be saved and applied with `git apply`.

Answers stream in word by word as the model writes them. For `fix`, the
explanation streams and the diff is shown when the model finishes. Use
`turbo --no-stream ...` to print answers all at once. In `chat`, Ctrl+C stops
the current answer without leaving the chat.

Exit codes: `0` ok, `1` model/Ollama error, `2` bad input.

## VS Code

1. **Start the server** (keep it running):
   ```
   turbo serve
   ```
   Or in VS Code run **TURBO: Start Server**, which runs `turbo serve` in a terminal.

2. **Install the extension**. Build it from a clone of this repo (no Node.js needed):
   ```
   python scripts/build_vsix.py
   code --install-extension dist/turbo-ai-local-0.2.0.vsix --force
   ```
   Then reload VS Code (*Developer: Reload Window*).

3. **Use it**: select code, then right-click, or use the Command Palette or a shortcut:

   | Command | Shortcut | |
   |---|---|---|
   | TURBO: Ask AI | `Ctrl+Alt+A` | Ask anything. A selection is attached as context. |
   | TURBO: Explain Selection | `Ctrl+Alt+E` | Explains the selection (or the whole file). |
   | TURBO: Fix Selection | `Ctrl+Alt+F` | Explains the bug and opens a diff. Click ✓ to accept or ✕ to reject. `Ctrl+Z` undoes. |
   | TURBO: Reset Conversation | | Clears this workspace's history. |

   Answers appear in the **TURBO** side panel. The status bar item shows
   whether the server is reachable.

Settings (`Ctrl+,`, search "turbo"): `turbo.serverUrl`, `turbo.token`,
`turbo.projectPath`, `turbo.contextLines`, `turbo.requestTimeoutSeconds`.

## How fixes work

The model is asked for an explanation plus the **complete corrected code**.
TURBO then builds the unified diff itself with Python's `difflib`, so the
diff is always valid and applies cleanly, even with small models that write
bad diffs by hand. Indentation, CRLF line endings and the trailing newline of
the selection are preserved, and hunk line numbers match the real file. If the
model's answer looks truncated, you get a warning.

Small local models can still be wrong. **Read the diff before accepting.**

## Tests

```
.venv\Scripts\python -m unittest discover -s tests -t .
```

Extension helpers (uses VS Code's bundled Node, no install needed):
```
set ELECTRON_RUN_AS_NODE=1
"%LOCALAPPDATA%\Programs\Microsoft VS Code\Code.exe" --test vscode-extension\test\lib.test.js
```

The `test_*.py` files in the project root are older hands-on scripts (microphone,
voice, Google Drive). They are not part of the test suite.

## Server API

`turbo serve` listens on `127.0.0.1:8765` and only accepts requests from local
programs. Web pages are refused. Set `TURBO_TOKEN` to also require an
`X-Turbo-Token` header.

| Endpoint | Body |
|---|---|
| `GET /health` | - |
| `POST /ask` | `{question, session_id?, context?, language?}` → `{answer}` |
| `POST /explain` | `{code, filename?, language?, question?, context?, session_id?}` → `{answer}` |
| `POST /fix` | `{code, filename?, language?, instruction?, context?, session_id?, start_line?}` → `{explanation, diff, fixed_code, changed, warnings}` |
| `POST /reset` | `{session_id?}` → `{cleared}` |

## Troubleshooting

- **"Can't reach Ollama"**: start the Ollama app, or run `ollama serve`.
- **"Model ... is not installed"**: run the `ollama pull ...` command shown.
- **VS Code says the server isn't running**: run `turbo serve`, or click *Start Server*.
- **Slow answers**: on a laptop without a GPU, a 7B model takes 30 seconds to a few
  minutes. Select less code, or set `TURBO_CODE_MODEL=qwen2.5-coder:3b`.
- **Port in use**: set `TURBO_PORT` and the matching `turbo.serverUrl`.

## Project layout

```
turbo_core/cli.py        the `turbo` command (turbo.py / turbo.cmd run it from a clone)
pyproject.toml           packaging (pip/pipx install)
turbo_core/              config, Ollama client, prompts, diffing, history, assistant, server
vscode-extension/        VS Code extension (plain JavaScript)
scripts/build_vsix.py    packages the extension
tests/                   unittest suite (mocked Ollama)
turbo_voice.py etc.      original voice assistant (unchanged)
```

## License

MIT. See [LICENSE](LICENSE).
