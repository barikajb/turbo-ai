"""TURBO Manager: an agent that diagnoses and manages TURBO and Ollama.

It talks to a local model with Ollama tool calling. Tools that only look at
things run immediately. Tools that change anything (switch or download a
model, edit a file, run a command, change a setting) show exactly what they
will do and run only after the user answers yes.

File access is limited to TURBO's own folders (the installed package or the
source checkout, and ~/.turbo).
"""

from __future__ import annotations

import difflib
import importlib.metadata
import json
import platform
import re
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from . import __version__, system
from .config import save_setting, turbo_home
from .llm import LLMError

MAX_STEPS = 8
MAX_TOOL_OUTPUT = 4000
MAX_FILE_LINES = 200
BLOCKED_NAMES = {"credentials.json", "token.json"}

# Settings the manager may change, and whether they apply without a restart.
SETTINGS = {
    "TURBO_CHAT_MODEL": "now", "TURBO_CODE_MODEL": "now", "TURBO_MAX_TOKENS": "now",
    "TURBO_HISTORY_CHARS": "now", "TURBO_KEEP_ALIVE": "now", "TURBO_CONTEXT_TOKENS": "now",
    "TURBO_MAX_CONTEXT_TOKENS": "restart", "TURBO_TIMEOUT": "restart", "TURBO_OLLAMA_URL": "restart",
    "TURBO_PORT": "restart", "TURBO_HOST": "restart",
}


def package_root() -> Path:
    """The source checkout (if running from one) or the installed package folder."""
    here = Path(__file__).resolve().parent
    return here.parent if (here.parent / "pyproject.toml").is_file() else here


def allowed_roots() -> list[Path]:
    return [package_root(), turbo_home().resolve()]


def check_path(path: str) -> Path:
    """Resolve a path the model gave us; refuse anything outside TURBO's folders."""
    raw = Path(path).expanduser()
    resolved = (raw if raw.is_absolute() else package_root() / raw).resolve()
    for root in allowed_roots():
        try:
            relative = resolved.relative_to(root)
        except ValueError:
            continue
        if resolved.name in BLOCKED_NAMES or ".git" in relative.parts:
            raise PermissionError(f"{resolved.name} is off limits (it may hold secrets or git data).")
        return resolved
    roots = ", ".join(str(r) for r in allowed_roots())
    raise PermissionError(f"{resolved} is outside TURBO's folders ({roots}).")


def _clip(text: str, limit: int = MAX_TOOL_OUTPUT) -> str:
    return text if len(text) <= limit else text[:limit] + f"\n... ({len(text) - limit} more characters cut)"


def tool(name: str, description: str, /, *, changes: bool, **params: tuple[str, str, bool]):
    """Declare a tool. params: param_name=(json_type, description, required).

    name/description are positional-only so tools can have a parameter called 'name'.
    """
    def decorate(func):
        func.tool_spec = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": {p: {"type": t, "description": d} for p, (t, d, _) in params.items()},
                    "required": [p for p, (_, _, req) in params.items() if req],
                },
            },
        }
        func.tool_name = name
        func.changes = changes
        return func
    return decorate


class Manager:
    def __init__(self, assistant, model: str | None = None, max_steps: int = MAX_STEPS):
        self.assistant = assistant
        self.model_override = model
        self.max_steps = max_steps
        self.ui = None
        self._tool_support: dict[str, bool | None] = {}
        self.tools: dict[str, Callable] = {}
        # Look at the class, not the instance, so properties (like `model`,
        # which asks Ollama) aren't evaluated just to find the tools.
        for attr, value in vars(type(self)).items():
            if hasattr(value, "tool_spec"):
                self.tools[value.tool_name] = getattr(self, attr)

    @property
    def model(self) -> str:
        """The chat model (usually already loaded) if it can call tools,
        otherwise the code model."""
        if self.model_override:
            return self.model_override
        settings = self.assistant.settings
        check = getattr(self.assistant.llm, "supports_tools", None)
        if check is None:
            return settings.chat_model
        key = settings.chat_model
        if key not in self._tool_support:
            self._tool_support[key] = check(key)
        return settings.code_model if self._tool_support[key] is False else settings.chat_model

    # ---- entry point ------------------------------------------------------------------

    def run(self, task: str, ui) -> str:
        """Work on a task. Prints progress via ui; returns the final answer.

        Common requests (speed, setup check) run built-in playbooks: fast and
        predictable even with a small model. Everything else goes to the
        tool-calling agent.
        """
        self.ui = ui
        self.task = task
        try:
            intent = classify(task)
            if intent == "speed":
                return self.speed_playbook()
            if intent == "diagnose":
                return self.diagnose_playbook()
            return self.agent(task)
        except KeyboardInterrupt:
            ui.note("(stopped)")
            return ""

    # ---- playbooks ---------------------------------------------------------------------

    def speed_playbook(self) -> str:
        ui = self.ui
        status = self.call_tool("get_status", {})
        snap_loaded = {m["name"] for m in status.get("loaded_models", [])}
        installed = [m["name"] for m in status.get("installed_models", [])]
        details = [{"name": m["name"], "parameters": m["parameters"]} for m in status.get("installed_models", [])]
        settings = self.assistant.settings
        done: list[str] = []

        if not status.get("ollama_running"):
            return self.finish("Ollama isn't running, so nothing can be sped up yet. Start the Ollama "
                               "app (or run `ollama serve`), then try again.")

        # 1. A smaller model for each role that is using a big one.
        for role, current in (("code", settings.code_model), ("chat", settings.chat_model)):
            size = system.model_billions(current, details)
            if size is not None and size <= 3.5:
                continue
            candidates = [name for name, _ in system.FAST_MODELS[role]]
            choice = next((c for c in candidates if c in installed), None)
            if choice is None:
                choice = candidates[0]
                result = self.call_tool("pull_model", {"name": choice})
                if not result.get("ok"):
                    continue
            result = self.call_tool("switch_model", {"role": role, "name": choice, "save": self.wants_save()})
            if result.get("ok"):
                done.append(f"{role} model: {current} → {choice}")

        # 2. Free RAM held by models TURBO isn't using.
        active = {self.assistant.settings.chat_model, self.assistant.settings.code_model}
        ram_free = status.get("ram_free_gb")
        for name in sorted(snap_loaded - active):
            if ram_free is not None and ram_free < 4:
                if self.call_tool("unload_model", {"name": name}).get("ok"):
                    done.append(f"unloaded {name} from memory")

        # 3. A grown context window slows every answer on CPU.
        if status.get("context_tokens", 0) > 4096:
            if self.call_tool("change_setting", {"key": "TURBO_CONTEXT_TOKENS", "value": "4096"}).get("ok"):
                done.append("context window back to 4096 tokens")

        lines = []
        if done:
            lines.append("**Done:**\n" + "\n".join(f"- {d}" for d in done))
        else:
            lines.append("Your setup is already on the fast settings: both models are 3B or smaller.")
            if all((system.model_billions(m, details) or 0) > 2 for m in active):
                lines.append("- The next step down is `qwen2.5-coder:1.5b` (1.0 GB): faster, but "
                             "noticeably weaker at fixes. `/model code qwen2.5-coder:1.5b` to try it.")
        tips = [t for t in status.get("speed_tips", []) if "/model" not in t]
        if tips:
            lines.append("**Also worth knowing:**\n" + "\n".join(f"- {t}" for t in tips))
        return self.finish("\n\n".join(lines))

    def diagnose_playbook(self) -> str:
        diag = self.call_tool("run_diagnostics", {})
        status = self.call_tool("get_status", {})
        problems: list[str] = []
        ok: list[str] = []

        missing = [p for p, v in diag.get("dependencies", {}).items() if v == "NOT INSTALLED"]
        if missing:
            command = f'"{sys.executable}" -m pip install ' + " ".join(missing)
            result = self.call_tool("run_command", {"command": command,
                                                    "reason": f"TURBO needs {', '.join(missing)}."})
            (ok if result.get("exit_code") == 0 else problems).append(
                f"dependencies {', '.join(missing)}: " +
                ("installed" if result.get("exit_code") == 0 else "still missing - run: " + command))
        else:
            ok.append("Python dependencies are installed")

        if not diag.get("ollama_running"):
            problems.append("Ollama isn't running: start the Ollama app (or run `ollama serve`)")
        else:
            ok.append("Ollama is running")
            settings = self.assistant.settings
            for role, key in (("chat", "chat_model_installed"), ("code", "code_model_installed")):
                name = getattr(settings, f"{role}_model")
                if diag.get(key):
                    ok.append(f"{role} model {name} is installed")
                elif self.call_tool("pull_model", {"name": name}).get("ok"):
                    ok.append(f"{role} model {name} downloaded")
                else:
                    problems.append(f"{role} model {name} isn't installed: `ollama pull {name}`")

        problems.extend(status.get("speed_tips", []))
        lines = ["**Looks good:**\n" + "\n".join(f"- {o}" for o in ok)]
        if problems:
            lines.append("**Needs attention:**\n" + "\n".join(f"- {p}" for p in problems))
        else:
            lines.append("No problems found.")
        return self.finish("\n\n".join(lines))

    def finish(self, answer: str) -> str:
        self.ui.markdown(answer)
        return answer

    def wants_save(self) -> bool:
        return bool(re.search(r"\b(save|keep|default|permanent(ly)?|always|from now on)\b",
                              getattr(self, "task", ""), re.IGNORECASE))

    # ---- the tool-calling agent ------------------------------------------------------

    def agent(self, task: str) -> str:
        ui = self.ui
        llm = self.assistant.llm
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt()},
            {"role": "user", "content": task},
        ]
        specs = [func.tool_spec for func in self.tools.values()]
        seen: set[str] = set()

        for _ in range(self.max_steps):
            reply = ui.wait(lambda: llm.chat_tools(messages, self.model, specs), "Thinking")
            calls = extract_tool_calls(reply)
            if not calls:
                return self.finish((reply.get("content") or "").strip() or "(no answer)")

            messages.append({"role": "assistant", "content": reply.get("content") or "",
                             "tool_calls": [{"function": {"name": n, "arguments": a}} for n, a in calls]})
            repeated = False
            for name, args in calls:
                key = f"{name}:{json.dumps(args, sort_keys=True)}"
                if key in seen:
                    repeated = True
                    continue
                seen.add(key)
                result = self.call_tool(name, args)
                messages.append({"role": "tool", "tool_name": name, "name": name,
                                 "content": _clip(json.dumps(result, default=str))})
            if repeated:
                break  # small models loop; answer with what we have
        return self.summarize(messages)

    def summarize(self, messages: list[dict[str, Any]]) -> str:
        """Ask for a final answer without tools (after a loop or too many steps)."""
        messages = messages + [{"role": "user", "content":
                                "Stop using tools. Tell me in a few short sentences what you found and "
                                "what was changed (only what the tool results confirm)."}]
        reply = self.ui.wait(lambda: self.assistant.llm.chat_tools(messages, self.model, []), "Summing up")
        return self.finish((reply.get("content") or "").strip() or "(no answer)")

    def call_tool(self, name: str, args: dict[str, Any]) -> Any:
        func = self.tools.get(name)
        if func is None:
            return {"error": f"Unknown tool {name!r}. Available: {', '.join(self.tools)}"}
        if not isinstance(args, dict):
            return {"error": "Arguments must be a JSON object."}
        label = ", ".join(f"{k}={v!r}" for k, v in args.items() if k not in ("new_text", "old_text"))
        self.ui.note(f"  → {name}({label})")
        try:
            return func(**args)
        except TypeError as error:
            return {"error": f"Bad arguments for {name}: {error}"}
        except (PermissionError, ValueError, OSError, LLMError) as error:
            return {"error": str(error)}

    def system_prompt(self) -> str:
        settings = self.assistant.settings
        snap = system.snapshot(self.assistant)
        hardware = (f"{snap.os_name}, {snap.cpu_cores} CPU cores, "
                    f"{snap.ram_free_gb} GB free of {snap.ram_total_gb} GB RAM")
        gpu = ("no model loaded yet" if not snap.loaded else
               ", ".join(f"{m['name']} {m['gpu_percent']}% on GPU" for m in snap.loaded))
        fast = "; ".join(f"{role}: " + ", ".join(f"{n} ({s})" for n, s in models)
                         for role, models in system.FAST_MODELS.items())
        return f"""You are TURBO Manager, the system agent of TURBO (a local coding assistant).
You diagnose and manage TURBO and Ollama on this computer.

Facts (from TURBO):
- TURBO {__version__} at {package_root()}; data {turbo_home()}
- Python {platform.python_version()}: {sys.executable}
- {hardware}; in memory: {gpu}
- Active: chat {settings.chat_model}, code {settings.code_model}
- Fast small models: {fast}. 1.5-3B models are already the fast tier; never suggest bigger ones for speed.

Rules:
- Check real state with a tool before acting or advising. Never guess.
- Change only what the user asked for. For "too slow", switch only the slow role.
  Use save=true only if the user asks to keep it.
- Changes need the user's approval. If declined, accept it and suggest alternatives.
- Only claim something was done if a tool result confirms it.
- pip: "{sys.executable}" -m pip install <package>
- Finish with a short, precise answer (a few sentences or bullets)."""

    # ---- tools: look --------------------------------------------------------------

    @tool("get_status", "Models (installed, loaded, GPU share), RAM, CPU, speed tips. "
          "Use first for speed questions.", changes=False)
    def get_status(self) -> dict:
        snap = system.snapshot(self.assistant, cpu_sample_seconds=0.3)
        data = snap.to_dict()
        data["speed_tips"] = system.speed_tips(snap)
        return data

    @tool("run_diagnostics", "TURBO setup: Python, dependency versions, Ollama, models, paths, "
          "settings.", changes=False)
    def run_diagnostics(self) -> dict:
        deps = {}
        for package in ("requests", "rich", "prompt_toolkit", "psutil"):
            try:
                deps[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                deps[package] = "NOT INSTALLED"
        snap = system.snapshot(self.assistant)
        settings = self.assistant.settings
        return {
            "turbo_version": __version__,
            "python": f"{platform.python_version()} ({sys.executable})",
            "dependencies": deps,
            "ollama_running": snap.ollama_ok,
            "chat_model_installed": snap.is_installed(settings.chat_model) if snap.ollama_ok else None,
            "code_model_installed": snap.is_installed(settings.code_model) if snap.ollama_ok else None,
            "package_folder": str(package_root()),
            "data_folder": str(turbo_home()),
            "settings_file": str(turbo_home() / ".env"),
            "settings": {k: getattr(settings, k) for k in (
                "ollama_url", "chat_model", "code_model", "context_tokens", "max_context_tokens",
                "max_tokens", "keep_alive", "history_chars", "timeout_seconds", "port")},
        }

    @tool("list_files", "List a folder in TURBO's files.", changes=False,
          path=("string", "Folder (default '.')", False))
    def list_files(self, path: str = ".") -> dict:
        folder = check_path(path)
        if not folder.is_dir():
            return {"error": f"{folder} is not a folder."}
        entries = sorted(folder.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        return {"folder": str(folder), "entries": [
            (p.name + "/") if p.is_dir() else f"{p.name} ({p.stat().st_size} bytes)"
            for p in entries if p.name not in (".git", "__pycache__", ".venv")
        ][:200]}

    @tool("read_file", "Read a TURBO file with line numbers.", changes=False,
          path=("string", "File path", True),
          start_line=("integer", "First line", False),
          end_line=("integer", "Last line", False))
    def read_file(self, path: str, start_line: int = 1, end_line: int | None = None) -> dict:
        file = check_path(path)
        if not file.is_file():
            return {"error": f"{file} doesn't exist."}
        lines = file.read_text(encoding="utf-8", errors="replace").splitlines()
        start = max(1, int(start_line))
        end = min(len(lines), int(end_line) if end_line else start + MAX_FILE_LINES - 1,
                  start + MAX_FILE_LINES - 1)
        numbered = "\n".join(f"{n:>5}  {lines[n - 1]}" for n in range(start, end + 1))
        return {"file": str(file), "total_lines": len(lines), "shown": f"{start}-{end}", "text": numbered}

    # ---- tools: change (always ask first) -------------------------------------------

    @tool("switch_model", "Use an installed model for chat, code or both.", changes=True,
          role=("string", "chat, code or both", True),
          name=("string", "Model, e.g. qwen2.5-coder:3b", True),
          save=("boolean", "Keep as default (only if the user asked)", False))
    def switch_model(self, role: str, name: str, save: bool = False) -> dict:
        installed = self.assistant.llm.list_models()
        if name not in installed and f"{name}:latest" not in installed:
            return {"error": f"{name} isn't installed. Installed: {', '.join(installed)}. "
                             "Use pull_model to download it."}
        settings = self.assistant.settings
        role = role.lower()
        if role not in ("chat", "code", "both"):
            return {"error": "role must be chat, code or both"}
        targets = {"chat": [settings.chat_model], "code": [settings.code_model],
                   "both": [settings.chat_model, settings.code_model]}[role]
        if all(t == name for t in targets):
            return {"ok": True, "note": f"{role} already uses {name}; nothing to change."}
        # Small models like to set save=true on their own; only honor it if
        # the user actually asked to keep the change.
        save = bool(save) and self.wants_save()
        current = " / ".join(dict.fromkeys(targets))
        keep = " and save it as the default" if save else ""
        if not self.ui.confirm(f"Switch {role} model {current} → {name}{keep}?", default=True):
            return {"declined": True}
        message = system.switch_model(self.assistant, role, name, persist=bool(save))
        self.ui.warm(name)
        return {"ok": True, "result": message}

    @tool("pull_model", "Download an Ollama model.", changes=True,
          name=("string", "Model name", True))
    def pull_model(self, name: str) -> dict:
        if not re.fullmatch(r"[\w.\-/]+(:[\w.\-]+)?", name):
            return {"error": f"{name!r} doesn't look like an Ollama model name."}
        sizes = {n: s for models in system.FAST_MODELS.values() for n, s in models}
        size = f" ({sizes[name]})" if name in sizes else ""
        if not self.ui.confirm(f"Download {name}{size} with Ollama?", default=True):
            return {"declined": True}
        ok = self.ui.pull(name)
        return {"ok": ok} if ok else {"error": "Download didn't finish."}

    @tool("unload_model", "Free RAM by unloading a model (stays installed).", changes=True,
          name=("string", "Model name", True))
    def unload_model(self, name: str) -> dict:
        if not self.ui.confirm(f"Unload {name} from memory?", default=True):
            return {"declined": True}
        self.assistant.llm.unload(name)
        return {"ok": True}

    @tool("change_setting", "Save a TURBO setting. Keys: " + ", ".join(SETTINGS), changes=True,
          key=("string", "Setting key", True),
          value=("string", "Value", True))
    def change_setting(self, key: str, value: str) -> dict:
        key = key.strip().upper()
        if key not in SETTINGS:
            return {"error": f"{key} can't be changed here. Allowed: {', '.join(SETTINGS)}"}
        value = str(value).strip()
        numeric = {"TURBO_MAX_TOKENS", "TURBO_HISTORY_CHARS", "TURBO_CONTEXT_TOKENS",
                   "TURBO_MAX_CONTEXT_TOKENS", "TURBO_TIMEOUT", "TURBO_PORT"}
        if key in numeric and not value.isdigit():
            return {"error": f"{key} must be a whole number."}
        if not self.ui.confirm(f"Set {key}={value} (saved in {turbo_home() / '.env'})?", default=True):
            return {"declined": True}
        path = save_setting(key, value)
        applied = self.apply_live(key, value) if SETTINGS[key] == "now" else False
        return {"ok": True, "saved_to": str(path),
                "applies": "now" if applied else "next time TURBO starts"}

    def apply_live(self, key: str, value: str) -> bool:
        fields = {"TURBO_CHAT_MODEL": "chat_model", "TURBO_CODE_MODEL": "code_model",
                  "TURBO_MAX_TOKENS": "max_tokens", "TURBO_HISTORY_CHARS": "history_chars",
                  "TURBO_KEEP_ALIVE": "keep_alive", "TURBO_CONTEXT_TOKENS": "context_tokens"}
        field_name = fields[key]
        typed: Any = int(value) if value.isdigit() and field_name not in ("chat_model", "code_model",
                                                                           "keep_alive") else value
        self.assistant.settings = replace(self.assistant.settings, **{field_name: typed})
        llm = self.assistant.llm
        if field_name == "keep_alive" and hasattr(llm, "keep_alive"):
            llm.keep_alive = value
        if field_name == "context_tokens" and hasattr(llm, "context_tokens"):
            llm.context_tokens = typed  # explicit request: allowed to shrink
        return True

    @tool("edit_file", "Replace text in a TURBO file (old_text must match exactly once).", changes=True,
          path=("string", "File path", True),
          old_text=("string", "Exact current text, without line numbers", True),
          new_text=("string", "Replacement", True))
    def edit_file(self, path: str, old_text: str, new_text: str) -> dict:
        file = check_path(path)
        if not file.is_file():
            return {"error": f"{file} doesn't exist."}
        original = file.read_text(encoding="utf-8")
        count = original.count(old_text) if old_text else 0
        if count != 1:
            return {"error": f"old_text must appear exactly once in the file; found {count} times. "
                             "Read the file again and copy the text exactly."}
        updated = original.replace(old_text, new_text, 1)
        diff = "".join(difflib.unified_diff(
            original.splitlines(keepends=True), updated.splitlines(keepends=True),
            fromfile=f"a/{file.name}", tofile=f"b/{file.name}"))
        self.ui.show_diff(diff)
        if not self.ui.confirm(f"Apply this change to {file}?"):
            return {"declined": True}
        file.write_text(updated, encoding="utf-8")
        return {"ok": True, "file": str(file)}

    @tool("run_command", "Run a shell command (pip install, ollama, tests).", changes=True,
          command=("string", "Command line", True),
          reason=("string", "Why, in one sentence", True))
    def run_command(self, command: str, reason: str = "") -> dict:
        command = command.strip()
        if not command:
            return {"error": "Empty command."}
        if reason:
            self.ui.note(f"  Why: {reason}")
        if not self.ui.confirm(f"Run this command?\n    {command}\n ", default=False):
            return {"declined": True}
        try:
            done = self.ui.wait(lambda: subprocess.run(
                command, shell=True, cwd=str(package_root()), capture_output=True,
                text=True, encoding="utf-8", errors="replace", timeout=600,
            ), "Running")
        except subprocess.TimeoutExpired:
            return {"error": "The command took longer than 10 minutes and was stopped."}
        except KeyboardInterrupt:
            return {"error": "Stopped by the user."}
        output = (done.stdout or "") + (("\n[stderr]\n" + done.stderr) if done.stderr else "")
        self.ui.show_output(_clip(output, 1500))
        return {"exit_code": done.returncode, "output": _clip(output)}


SPEED_WORDS = re.compile(r"\b(slow(er|ly)?|fast(er)?|speed( ?up)?|sluggish|laggy|lag|quick(er)?|"
                         r"performance|optimi[sz]e)\b", re.IGNORECASE)
DIAGNOSE_WORDS = re.compile(r"\b(check|diagnos\w*|health|doctor|status|setup|what'?s wrong|"
                            r"is (everything|it) (ok|working))\b", re.IGNORECASE)


def classify(task: str) -> str:
    """'speed', 'diagnose' or 'agent'. Specific requests (naming a file,
    dependency, command or model) always go to the agent."""
    specific = (r"\b(files?|dependenc\w*|packages?|pip|install\w*|errors?|traceback|exception|"
                r"edit|commands?|settings?|config\w*|unload|code ?base|import)\b"
                r"|[\w.-]+:[\w.]+"     # a model tag like qwen2.5-coder:1.5b
                r"|\.\w{1,4}\b")       # a file name like turbo.py
    if re.search(specific, task, re.IGNORECASE):
        return "agent"
    if SPEED_WORDS.search(task):
        return "speed"
    if DIAGNOSE_WORDS.search(task):
        return "diagnose"
    return "agent"


def extract_tool_calls(message: dict[str, Any]) -> list[tuple[str, dict]]:
    """Tool calls from a model reply.

    Ollama normally returns them in message['tool_calls']. Small models
    sometimes write the call as JSON in the text instead, so that is
    accepted too.
    """
    calls = []
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        args = function.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                args = {}
        if function.get("name"):
            calls.append((function["name"], args))
    if calls:
        return calls

    content = (message.get("content") or "").strip()
    match = re.fullmatch(r"(?:```(?:json)?\s*)?(\{.*\})(?:\s*```)?", content, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1))
        except ValueError:
            return []
        name = data.get("name") or (data.get("function") or {}).get("name")
        args = data.get("arguments") or data.get("parameters") or {}
        if isinstance(name, str) and isinstance(args, dict):
            return [(name, args)]
    return []
