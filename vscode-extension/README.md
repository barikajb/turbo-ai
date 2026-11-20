# TURBO AI (local)

VS Code commands for your local TURBO assistant. Requests go only to the TURBO
server on your machine, which talks to Ollama on your machine.

| Command | Shortcut | What it does |
|---|---|---|
| TURBO: Ask AI | Ctrl+Alt+A | Ask anything. If code is selected, it's attached. Follow-ups remember the conversation. |
| TURBO: Explain Selection | Ctrl+Alt+E | Explains the selected lines (or the whole file if nothing is selected). |
| TURBO: Fix Selection | Ctrl+Alt+F | Explains the bug and opens a diff of the proposed fix. Accept with ✓, reject with ✕. |
| TURBO: Reset Conversation | | Forget this workspace's conversation. |
| TURBO: Start Server | | Runs `turbo.py serve` in a terminal. |

Ask / Explain / Fix are also in the editor right-click menu.

Start the server first: `turbo serve` in the TURBO folder. See the main README.
