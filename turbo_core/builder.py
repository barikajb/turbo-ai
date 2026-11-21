"""Build mode: describe an app or script and TURBO writes it to disk.

The model works in a loop with four tools (write_file, read_file,
list_directory, run_shell_command) until the project is done. Everything is
scoped to one workspace folder:

- File paths can't leave the workspace.
- Overwriting an existing file and running a command ask [Y/n] first, unless
  /auto is on.
- Commands run inside the workspace with a time limit; a few clearly
  destructive commands are always refused.
- To stay within free per-minute token limits, the contents of files that are
  already saved are replaced in the model's history by a short note (the
  model can read_file them again when it needs to).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from .llm import LLMError
from .manager import extract_tool_calls, tool

MAX_STEPS = 40
MAX_READ_LINES = 400
MAX_OUTPUT_CHARS = 3000
COMMAND_TIMEOUT = 90
SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", ".mypy_cache", ".pytest_cache",
             "dist", "build", ".next"}

# Refused even in /auto: things that can damage the computer outside the project.
DANGEROUS = re.compile(
    r"\b(format|diskpart|shutdown|reboot|mkfs|reg\s+delete|bcdedit|cipher\s+/w|takeown|icacls)\b"
    r"|\brm\s+-[a-z]*r[a-z]*f?\s+(/|~|\*|\.\.|[a-z]:)"
    r"|\b(rd|rmdir|del|erase)\b[^\n]*\s/s\b[^\n]*([a-z]:\\|\\\\|\.\.)"
    r"|remove-item[^\n]*-recurse[^\n]*([a-z]:\\|\.\.|~)"
    r"|\|\s*(sh|bash|iex|invoke-expression)\b"
    r"|:\(\)\s*\{",
    re.IGNORECASE,
)

BUILD_PROMPT = """You are TURBO in build mode: you create complete, working software projects on the
user's disk using tools. The project folder is the current directory for every tool;
always use relative paths like "src/main.py".

How to work:
1. Plan briefly: decide the files you need (keep it as small as the request allows).
2. Create every file with write_file, with the COMPLETE file content. Never put code in
   your chat reply instead of writing the file, and never write placeholder code like
   "# TODO" or "...": every file must work as written.
3. Include what's needed to run it: e.g. requirements.txt or package.json, and a short
   README.md with run instructions.
4. Check your work: run quick, non-interactive checks with run_shell_command, e.g.
   "python -m py_compile app.py", running a test, or a short script. Never start servers
   or anything that waits for input or runs forever.
5. If a check fails, read the error, fix the file with write_file, and check again.
6. When everything is done, reply with a short summary: the files, how to run it, and
   anything the user must do (e.g. install packages). Don't repeat the code.

Environment: {os_name}, Python at "{python}". Use "{python}" for Python commands.
Prefer the Python standard library unless the user asks for a framework."""


class BuildError(ValueError):
    pass


class Builder:
    """One project workspace, with its own build conversation."""

    def __init__(self, assistant, workspace: Path, ui, auto: bool = False, max_steps: int = MAX_STEPS):
        self.assistant = assistant
        self.workspace = Path(workspace).resolve()
        self.ui = ui
        self.auto = auto
        self.max_steps = max_steps
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": BUILD_PROMPT.format(
            os_name="Windows" if os.name == "nt" else sys.platform, python=sys.executable)}]
        self.created: list[str] = []
        self.updated: list[str] = []
        self.tools: dict[str, Callable] = {}
        for attr, value in vars(type(self)).items():
            if hasattr(value, "tool_spec"):
                self.tools[value.tool_name] = getattr(self, attr)

    @property
    def model(self) -> str:
        return self.assistant.settings.code_model

    # ---- the agent loop ------------------------------------------------------------

    def run(self, request: str) -> str:
        """Build (or keep building) from a plain-English request."""
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.created, self.updated = [], []
        existing = self.tree(max_entries=60)
        note = f"\n\nThe project folder already contains:\n{existing}" if existing else \
            "\n\nThe project folder is empty."
        self.messages.append({"role": "user", "content": request + note})
        specs = [func.tool_spec for func in self.tools.values()]
        llm = self.assistant.llm
        repeats = 0
        last_key = None

        for step in range(self.max_steps):
            reply = self.ui.with_choices(lambda: self.ui.wait(
                lambda: llm.chat_tools(self.messages, self.model, specs, temperature=0.2, max_tokens=4096),
                "Building" if step else "Planning"))
            calls = extract_tool_calls(reply)
            if not calls:
                answer = (reply.get("content") or "").strip() or "Done."
                self.messages.append({"role": "assistant", "content": answer})
                missing = self.missing_files(answer)
                if missing:
                    answer += ("\n\n**Note from TURBO:** the summary mentions "
                               + ", ".join(f"`{m}`" for m in missing)
                               + " but no such file exists in the project. Ask for it with /build if you need it.")
                return answer

            self.messages.append({"role": "assistant", "content": reply.get("content") or "",
                                  "tool_calls": [{"function": {"name": n, "arguments": a}} for n, a in calls]})
            for name, args in calls:
                key = f"{name}:{json.dumps(args, sort_keys=True)[:500]}"
                repeats = repeats + 1 if key == last_key else 0
                last_key = key
                if repeats >= 2:
                    result = {"error": "You repeated the same call. Use the result you already have."}
                else:
                    result = self.call_tool(name, args)
                self.messages.append({"role": "tool", "tool_name": name, "name": name,
                                      "content": _clip(json.dumps(result, default=str), MAX_OUTPUT_CHARS)})
            self.compact()
        return "Stopped after the maximum number of steps. Run /build again to continue."

    def call_tool(self, name: str, args: dict[str, Any]) -> Any:
        func = self.tools.get(name)
        if func is None:
            return {"error": f"Unknown tool {name!r}. Available: {', '.join(self.tools)}"}
        if not isinstance(args, dict):
            return {"error": "Arguments must be a JSON object."}
        try:
            return func(**args)
        except TypeError as error:
            return {"error": f"Bad arguments for {name}: {error}"}
        except (BuildError, OSError, LLMError, UnicodeDecodeError) as error:
            return {"error": str(error)}

    def missing_files(self, summary: str) -> list[str]:
        """File names the model's summary mentions that aren't in the project.
        (Small models sometimes describe files they never wrote.)"""
        names = set(re.findall(
            r"(?<![\w/.-])((?:[\w-]+/)*[\w.-]+\.(?:py|md|txt|json|js|ts|tsx|jsx|html|css|toml|ya?ml|cfg|ini|sh|ps1))\b",
            summary))
        missing = []
        for name in sorted(names):
            candidates = [self.workspace / name] + list(self.workspace.rglob(Path(name).name))
            if not any(c.is_file() for c in candidates):
                missing.append(name)
        return missing

    def compact(self) -> None:
        """Replace contents of already-saved files in the history with a short
        note, so every step doesn't re-send whole files (free token limits)."""
        for message in self.messages:
            for call in message.get("tool_calls") or []:
                function = call.get("function", {})
                args = function.get("arguments")
                if function.get("name") == "write_file" and isinstance(args, dict) \
                        and isinstance(args.get("content"), str) and not args.get("_compacted"):
                    text = args["content"]
                    lines = text.count("\n") + (0 if text.endswith("\n") else 1)
                    function["arguments"] = {"filepath": args.get("filepath"), "_compacted": True,
                                             "content": f"<{lines} lines saved to disk; use read_file to see them>"}

    # ---- paths -------------------------------------------------------------------------

    def resolve(self, filepath: str) -> Path:
        if not isinstance(filepath, str) or not filepath.strip():
            raise BuildError("filepath is empty.")
        raw = Path(filepath.strip().strip('"'))
        target = (raw if raw.is_absolute() else self.workspace / raw).resolve()
        try:
            relative = target.relative_to(self.workspace)
        except ValueError:
            raise BuildError(f"{filepath} is outside the project folder ({self.workspace}). "
                             "Use a relative path.") from None
        if ".git" in relative.parts:
            raise BuildError("Files inside .git can't be changed.")
        return target

    def rel(self, path: Path) -> str:
        return path.relative_to(self.workspace).as_posix() or "."

    def tree(self, folder: Path | None = None, depth: int = 3, max_entries: int = 200) -> str:
        folder = folder or self.workspace
        if not folder.is_dir():
            return ""
        lines: list[str] = []

        def walk(path: Path, level: int) -> None:
            if level > depth or len(lines) >= max_entries:
                return
            for entry in sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name.lower())):
                if len(lines) >= max_entries:
                    lines.append("  " * level + "...")
                    return
                if entry.name in SKIP_DIRS:
                    continue
                if entry.is_dir():
                    lines.append("  " * level + entry.name + "/")
                    walk(entry, level + 1)
                else:
                    lines.append("  " * level + f"{entry.name} ({entry.stat().st_size} bytes)")

        walk(folder, 0)
        return "\n".join(lines)

    # ---- tools ---------------------------------------------------------------------------

    @tool("write_file", "Create or replace a file with its COMPLETE content (folders are created).",
          changes=True,
          filepath=("string", "Relative path, e.g. src/main.py", True),
          content=("string", "The complete file content", True))
    def write_file(self, filepath: str, content: str) -> dict:
        target = self.resolve(filepath)
        if not isinstance(content, str) or not content.strip():
            raise BuildError("Refusing to write an empty file. Provide the complete content.")
        rel = self.rel(target)
        exists = target.exists()
        if exists and target.is_dir():
            raise BuildError(f"{rel} is a folder.")
        if exists:
            old = target.read_text(encoding="utf-8", errors="replace")
            if old == content:
                return {"ok": True, "note": f"{rel} already has exactly this content."}
            if not self.auto and rel not in self.created and rel not in self.updated:
                if not self.ui.confirm(f"Overwrite existing {rel}?", default=True):
                    return {"declined": True, "note": "The user kept the existing file."}
        target.parent.mkdir(parents=True, exist_ok=True)
        if not content.endswith("\n"):
            content += "\n"
        target.write_text(content, encoding="utf-8")
        lines = content.count("\n")
        if exists:
            if rel not in self.created and rel not in self.updated:
                self.updated.append(rel)
            self.ui.file_event("Updated", rel, lines)
        else:
            self.created.append(rel)
            self.ui.file_event("Created", rel, lines)
        return {"ok": True, "file": rel, "lines": lines}

    @tool("read_file", "Read a file in the project, with line numbers.", changes=False,
          filepath=("string", "Relative path", True))
    def read_file(self, filepath: str) -> dict:
        target = self.resolve(filepath)
        if not target.is_file():
            raise BuildError(f"{filepath} doesn't exist.")
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        shown = lines[:MAX_READ_LINES]
        text = "\n".join(f"{n:>4}  {line}" for n, line in enumerate(shown, 1))
        more = f"\n... ({len(lines) - MAX_READ_LINES} more lines)" if len(lines) > MAX_READ_LINES else ""
        return {"file": self.rel(target), "lines": len(lines), "text": text + more}

    @tool("list_directory", "Show the project's folders and files.", changes=False,
          path=("string", "Relative folder (default '.')", False))
    def list_directory(self, path: str = ".") -> dict:
        folder = self.resolve(path or ".")
        if not folder.is_dir():
            raise BuildError(f"{path} is not a folder.")
        return {"folder": self.rel(folder), "tree": self.tree(folder) or "(empty)"}

    @tool("run_shell_command", "Run a quick, non-interactive command in the project folder "
          "(e.g. tests or py_compile). Never start servers.", changes=True,
          command=("string", "Command line", True))
    def run_shell_command(self, command: str) -> dict:
        command = (command or "").strip()
        if not command:
            raise BuildError("Empty command.")
        if DANGEROUS.search(command):
            self.ui.note(f"  ✗ Refused: {command}", style="red")
            return {"error": "Refused: this command could damage files outside the project."}
        if not self.auto and not self.ui.confirm(f"Run: {command}", default=True):
            return {"declined": True, "note": "The user chose not to run it."}
        result = self.ui.wait(lambda: run_command(command, self.workspace), f"Running {command[:50]}")
        mark = "✔" if result["exit_code"] == 0 else "✗"
        style = "green" if result["exit_code"] == 0 else "yellow"
        self.ui.note(f"  {mark} Ran {command} (exit {result['exit_code']})" +
                     (" - stopped: took too long" if result.get("timed_out") else ""), style=style)
        return result


def run_command(command: str, cwd: Path, timeout: int = COMMAND_TIMEOUT) -> dict:
    """Run a shell command; on timeout, stop it and everything it started."""
    process = subprocess.Popen(
        command, shell=True, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace",
    )
    try:
        output, _ = process.communicate(timeout=timeout)
        timed_out = False
    except subprocess.TimeoutExpired:
        _kill_tree(process.pid)
        output, _ = process.communicate()
        timed_out = True
    output = output or ""
    return {"exit_code": process.returncode if not timed_out else -1, "timed_out": timed_out,
            "output": _clip(output, MAX_OUTPUT_CHARS) + (
                f"\n[stopped after {timeout}s: commands must finish on their own]" if timed_out else "")}


def _kill_tree(pid: int) -> None:
    try:
        import psutil

        parent = psutil.Process(pid)
        for child in parent.children(recursive=True):
            child.kill()
        parent.kill()
    except Exception:  # noqa: BLE001 - best effort
        pass


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + f"\n... ({len(text) - limit} more characters cut)"


BUILD_REQUEST = re.compile(
    r"\b(create|build|make|scaffold|generate|write|set ?up|start)\b.{0,60}?"
    r"\b(app|application|project|script|website|web ?site|api|cli|game|tool|bot|server|program|"
    r"extension|dashboard|page|calculator|scraper)\b",
    re.IGNORECASE)
FOLDER_HINT = re.compile(r"\b(?:in|into|inside|under|at|called|named)\s+[\"']?(\.{0,2}[/\\]?[\w.\-]+(?:[/\\][\w.\-]+)*)",
                         re.IGNORECASE)


def looks_like_build_request(text: str) -> bool:
    return bool(BUILD_REQUEST.search(text)) and not text.strip().endswith("?")


def suggest_folder(text: str) -> str:
    """Folder name from 'in ./my-app' / 'called notes-api', else from the request."""
    match = FOLDER_HINT.search(text)
    if match:
        candidate = match.group(1).strip("/\\")
        if candidate and candidate.lower() not in {"python", "javascript", "flask", "react", "node", "a", "the"}:
            return candidate.lstrip("./\\") or "new-project"
    words = re.findall(r"[a-z0-9]+", text.lower())
    stop = {"create", "build", "make", "a", "an", "the", "new", "simple", "small", "me", "please", "that",
            "with", "for", "in", "and", "to", "of", "using", "python", "app", "script", "write", "generate"}
    useful = [w for w in words if w not in stop][:3]
    return "-".join(useful) or "new-project"
