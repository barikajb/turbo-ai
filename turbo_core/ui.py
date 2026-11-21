"""Interactive terminal UI for `turbo` (rich + prompt_toolkit).

Only used on a real terminal. Pipes and tests use the plain loop in cli.py.
"""

from __future__ import annotations

import html
import queue
import threading
import time
from typing import Any, Callable

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.progress import BarColumn, DownloadColumn, Progress, TextColumn, TransferSpeedColumn
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from . import __version__, system
from .assistant import Assistant, InputError
from .config import turbo_home
from .llm import LLMError

BANNER = [
    "████████╗██╗   ██╗██████╗ ██████╗  ██████╗ ",
    "╚══██╔══╝██║   ██║██╔══██╗██╔══██╗██╔═══██╗",
    "   ██║   ██║   ██║██████╔╝██████╔╝██║   ██║",
    "   ██║   ██║   ██║██╔══██╗██╔══██╗██║   ██║",
    "   ██║   ╚██████╔╝██║  ██║██████╔╝╚██████╔╝",
    "   ╚═╝    ╚═════╝ ╚═╝  ╚═╝╚═════╝  ╚═════╝ ",
]
GRADIENT = ["#5fd7ff", "#5fafff", "#5f87ff", "#875fff", "#af5fff", "#d75fff"]
ACCENT = "#af87ff"
DIM = "grey50"

COMMANDS = {
    "/help": "show commands and keys",
    "/clear": "forget this conversation and clear the screen",
    "/model": "show or switch models: /model NAME  |  /model code NAME  |  add --save to keep it",
    "/models": "list installed models",
    "/file": "share a file with your next question: /file PATH [LINES]",
    "/paste": "paste a block of code (finish with /end)",
    "/status": "Ollama, models, memory and speed tips",
    "/manage": "ask TURBO Manager to diagnose, optimize or fix the setup",
    "/exit": "quit",
}
KEYS = {
    "Enter": "send",
    "Alt+Enter": "new line",
    "Ctrl+C": "stop the answer / clear the line / press twice to quit",
    "Ctrl+L": "clear the screen",
    "Up / Down": "previous messages",
    "Tab": "complete a /command",
}


class Cancelled(Exception):
    pass


# ---- streaming ------------------------------------------------------------------

class _StreamView:
    """Live-updating view: a spinner until the first words, then the answer."""

    def __init__(self, label: str):
        self.text = ""
        self.label = label
        self.start = time.monotonic()
        self.spinner = Spinner("dots", style=ACCENT)

    def __rich_console__(self, console, options):
        elapsed = time.monotonic() - self.start
        if not self.text.strip():
            self.spinner.update(text=Text(f" {self.label}... {elapsed:.0f}s", style=DIM))
            yield self.spinner
            return
        self.spinner.update(text=" ")  # rich ignores "", so use a space to drop the label
        # Show the newest part if the answer is taller than the screen; the
        # full answer is printed once it's done.
        room = max(5, options.max_height - 2)
        lines = self.text.splitlines()
        body = Markdown(self.text) if len(lines) <= room else Text("\n".join(lines[-room:]))
        yield Group(body, self.spinner)


def stream_call(console: Console, call: Callable[[Callable[[str], None]], Any], label: str = "Thinking"):
    """Run call(on_token) in a worker thread and render the answer live.

    Returns (result, text). result is None if the user pressed Ctrl+C.
    Errors raised by call are re-raised here.
    """
    events: queue.Queue = queue.Queue()
    cancel = threading.Event()
    outcome: dict[str, Any] = {}

    def on_token(piece: str) -> None:
        if cancel.is_set():
            raise Cancelled()
        events.put(("token", piece))

    def worker() -> None:
        try:
            outcome["result"] = call(on_token)
            events.put(("done", None))
        except Cancelled:
            events.put(("done", None))
        except BaseException as error:  # noqa: BLE001 - handed to the main thread
            events.put(("error", error))

    # The worker keeps the main thread free, so Ctrl+C works even while the
    # model is still reading the prompt (before the first word).
    threading.Thread(target=worker, daemon=True).start()
    view = _StreamView(label)
    try:
        with Live(view, console=console, refresh_per_second=12, transient=True):
            while True:
                try:
                    kind, payload = events.get(timeout=0.1)
                except queue.Empty:
                    continue
                if kind == "token":
                    view.text += payload
                elif kind == "error":
                    raise payload
                else:
                    break
    except KeyboardInterrupt:
        cancel.set()
        if view.text.strip():
            console.print(Markdown(view.text))
        console.print(Text("(stopped)", style=DIM))
        return None, view.text
    if view.text.strip():
        console.print(Markdown(view.text.strip()))
    return outcome.get("result"), view.text


# ---- prompt helpers ---------------------------------------------------------------

class SlashCompleter(Completer):
    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if not text.startswith("/") or " " in text:
            return
        for name, help_text in COMMANDS.items():
            if name.startswith(text):
                yield Completion(name, start_position=-len(text), display_meta=help_text)


def _key_bindings() -> KeyBindings:
    keys = KeyBindings()

    @keys.add("escape", "enter")
    def _newline(event):
        event.current_buffer.insert_text("\n")

    @keys.add("c-l")
    def _clear(event):
        event.app.renderer.clear()

    return keys


def format_size(size: int | float) -> str:
    return f"{size / system.GB:.1f} GB" if size else "?"


# ---- chat --------------------------------------------------------------------------

class ChatUI:
    def __init__(self, assistant: Assistant, session_id: str, console: Console | None = None,
                 manager=None):
        self.assistant = assistant
        self.session_id = session_id
        self.console = console or Console(highlight=False)
        self.manager = manager  # TURBO Manager (optional)
        self.pending_context: str | None = None
        self.pending_label = ""
        self.prompt: PromptSession | None = None

    # -- screen pieces

    def print_banner(self) -> None:
        console = self.console
        console.print()
        for line, color in zip(BANNER, GRADIENT):
            console.print(Text("  " + line, style=f"bold {color}"))
        console.print(Text(f"  local coding assistant · v{__version__}", style=DIM))
        console.print()
        self.print_status_line(system.snapshot(self.assistant))
        console.print(Text("  Ask anything, share code with /file, or type /help.  "
                           "Ctrl+C stops an answer.", style=DIM))
        console.print()

    def print_status_line(self, snap: system.Snapshot) -> None:
        line = Text("  ")
        if snap.ollama_ok:
            line.append("● ", style="green")
            line.append("Ollama running", style="green")
        else:
            line.append("● ", style="red")
            line.append("Ollama not reachable", style="red")
        line.append("   chat ", style=DIM)
        line.append(snap.chat_model, style="cyan")
        line.append("   code ", style=DIM)
        line.append(snap.code_model, style="cyan")
        if snap.ram_free_gb is not None:
            line.append(f"   RAM {snap.ram_free_gb} / {snap.ram_total_gb} GB free", style=DIM)
        self.console.print(line)

        warnings = []
        if snap.ollama_ok:
            for role, name in (("chat", snap.chat_model), ("code", snap.code_model)):
                if not snap.is_installed(name):
                    warnings.append(f"{role} model {name} isn't installed: /model to switch, "
                                    f"or run `ollama pull {name}`")
        tips = system.speed_tips(snap)
        if not snap.ollama_ok:
            warnings.extend(tips)
        else:
            # Only the urgent tip at startup; the rest are in /status.
            warnings.extend(t for t in tips if "RAM" in t or "spills" in t)
        for warning in warnings:
            self.console.print(Text(f"  ! {warning}", style="yellow"))

    def toolbar(self):
        settings = self.assistant.settings
        attached = f" · <style fg='ansiyellow'>📎 {html.escape(self.pending_label)}</style>" \
            if self.pending_label else ""
        return HTML(
            f" chat <style fg='ansicyan'>{html.escape(settings.chat_model)}</style>"
            f" · code <style fg='ansicyan'>{html.escape(settings.code_model)}</style>{attached}"
            " · /help · Ctrl+C stop · Ctrl+L clear"
        )

    def error(self, error: Exception | str) -> None:
        self.console.print(Text(f"✗ {error}", style="red"))

    def note(self, text: str, style: str = DIM) -> None:
        self.console.print(Text(text, style=style))

    def confirm(self, question: str, default: bool = False) -> bool:
        hint = "[Y/n]" if default else "[y/N]"
        try:
            answer = self.console.input(f"[yellow]? {question} {hint}[/yellow] ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            self.console.print()
            return False
        if not answer:
            return default
        return answer in ("y", "yes")

    # -- main loop

    def run(self) -> int:
        home = turbo_home()
        home.mkdir(parents=True, exist_ok=True)
        self.print_banner()
        self.warm(self.assistant.settings.chat_model)
        self.prompt = PromptSession(
            history=FileHistory(str(home / "history")),
            key_bindings=_key_bindings(),
            completer=SlashCompleter(),
            complete_while_typing=True,
            bottom_toolbar=self.toolbar,
        )
        last_interrupt = 0.0
        while True:
            try:
                line = self.prompt.prompt(HTML(f"<style fg='{ACCENT}'><b>›</b></style> "))
            except KeyboardInterrupt:
                now = time.monotonic()
                if now - last_interrupt < 2:
                    return 0
                last_interrupt = now
                self.note("Press Ctrl+C again to quit (or type /exit).")
                continue
            except EOFError:
                return 0
            line = line.strip()
            if not line:
                continue
            if line.startswith("/"):
                if self.handle_command(line) == "exit":
                    return 0
                continue
            self.ask(line)

    def warm(self, model: str) -> None:
        warm = getattr(self.assistant.llm, "warm", None)
        if warm:
            threading.Thread(target=warm, args=(model,), daemon=True).start()

    def ask(self, question: str) -> None:
        context = self.pending_context
        self.console.print(Text("◆ TURBO", style=f"bold {ACCENT}"))
        try:
            result, _ = stream_call(
                self.console,
                lambda on_token: self.assistant.ask(
                    question, session_id=self.session_id, context=context, on_token=on_token
                ),
            )
        except (LLMError, InputError) as error:
            self.error(error)
            return
        if result is not None:
            self.pending_context = None
            self.pending_label = ""
            if self.assistant.last_reply_truncated:
                self.note(f"(answer cut off at {self.assistant.settings.max_tokens} tokens: "
                          "ask \"continue\", or raise TURBO_MAX_TOKENS)")
        self.console.print()

    # -- slash commands

    def handle_command(self, line: str) -> str | None:
        name, _, rest = line.partition(" ")
        name = name.lower()
        rest = rest.strip()
        handlers = {
            "/help": self.cmd_help,
            "/clear": self.cmd_clear,
            "/reset": self.cmd_clear,
            "/model": self.cmd_model,
            "/models": self.cmd_models,
            "/file": self.cmd_file,
            "/paste": self.cmd_paste,
            "/status": self.cmd_status,
            "/manage": self.cmd_manage,
            "/exit": lambda _rest: "exit",
            "/quit": lambda _rest: "exit",
        }
        handler = handlers.get(name)
        if handler is None:
            self.error(f"Unknown command {name}. Type /help.")
            return None
        try:
            return handler(rest)
        except (LLMError, InputError, ValueError) as error:
            self.error(error)
            return None

    def cmd_help(self, _rest: str) -> None:
        table = Table.grid(padding=(0, 3))
        table.add_column(style="cyan", no_wrap=True)
        table.add_column(style=DIM)
        for name, text in COMMANDS.items():
            table.add_row(name, text)
        keys = Table.grid(padding=(0, 3))
        keys.add_column(style="cyan", no_wrap=True)
        keys.add_column(style=DIM)
        for key, text in KEYS.items():
            keys.add_row(key, text)
        self.console.print(Text("Commands", style="bold"))
        self.console.print(table)
        self.console.print()
        self.console.print(Text("Keys", style="bold"))
        self.console.print(keys)
        self.console.print()
        self.note("/manage examples: \"it's too slow\", \"check my setup\", "
                  "\"download a faster coding model\", \"fix a dependency issue\"")

    def cmd_clear(self, _rest: str) -> None:
        removed = self.assistant.reset(self.session_id)
        self.pending_context = None
        self.pending_label = ""
        self.console.clear()
        self.note(f"Conversation cleared ({removed} messages). Starting fresh.")

    def cmd_model(self, rest: str) -> None:
        args = rest.split()
        persist = "--save" in args
        args = [a for a in args if a != "--save"]
        settings = self.assistant.settings
        if not args:
            self.console.print(Text.assemble(("chat ", DIM), (settings.chat_model, "cyan"),
                                             ("   code ", DIM), (settings.code_model, "cyan")))
            self.note("Switch with /model NAME (both), /model chat NAME or /model code NAME. "
                      "Add --save to keep it. /models lists what's installed.")
            return
        role = "both"
        if args[0].lower() in ("chat", "code"):
            role = args[0].lower()
            args = args[1:]
        if len(args) != 1:
            raise ValueError("Usage: /model NAME  |  /model chat NAME  |  /model code NAME  [--save]")
        name = args[0]
        if not self.ensure_installed(name):
            return
        self.note(system.switch_model(self.assistant, role, name, persist), style="green")
        self.warm(name)

    def ensure_installed(self, name: str) -> bool:
        installed = self.assistant.llm.list_models()
        if name in installed or (":" not in name and f"{name}:latest" in installed):
            return True
        if not self.confirm(f"{name} isn't installed. Download it now?"):
            self.note("Not switched.")
            return False
        return self.pull(name)

    def pull(self, name: str) -> bool:
        columns = (TextColumn("  {task.description}"), BarColumn(), DownloadColumn(), TransferSpeedColumn())
        with Progress(*columns, console=self.console, transient=True) as progress:
            task = progress.add_task(f"Downloading {name}", total=None)

            def on_progress(status: str, completed: int, total: int) -> None:
                progress.update(task, description=f"{name}: {status}", completed=completed or 0,
                                total=total or None)

            try:
                self.assistant.llm.pull(name, on_progress)
            except KeyboardInterrupt:
                self.note("Download stopped. Run it again to resume.")
                return False
        self.note(f"✓ Downloaded {name}.", style="green")
        return True

    def cmd_models(self, _rest: str) -> None:
        snap = system.snapshot(self.assistant)
        if not snap.ollama_ok:
            raise LLMError("Ollama isn't running. Start the Ollama app (or run `ollama serve`).")
        loaded = {m["name"]: m for m in snap.loaded}
        table = Table(box=None, padding=(0, 2), header_style=DIM)
        for column in ("model", "size", "params", "in use as", "loaded"):
            table.add_column(column)
        for m in sorted(snap.installed, key=lambda m: m["size"]):
            roles = [r for r, n in (("chat", snap.chat_model), ("code", snap.code_model)) if n == m["name"]]
            live = loaded.get(m["name"])
            where = "" if not live else ("GPU" if live["gpu_percent"] == 100 else
                                         "CPU" if live["gpu_percent"] == 0 else f"{live['gpu_percent']}% GPU")
            table.add_row(m["name"], format_size(m["size"]), m["parameters"], ", ".join(roles), where,
                          style="cyan" if roles else None)
        self.console.print(table)
        self.note("Switch with /model NAME. Smaller models answer faster.")

    def cmd_status(self, _rest: str) -> None:
        snap = system.snapshot(self.assistant, cpu_sample_seconds=0.3)
        self.print_status_line(snap)
        if snap.cpu_percent is not None:
            self.note(f"  CPU {snap.cpu_percent:.0f}% busy · {snap.cpu_cores} cores · {snap.os_name}"
                      f" · context window {snap.context_tokens} tokens")
        for m in snap.loaded:
            where = "CPU" if m["gpu_percent"] == 0 else f"{m['gpu_percent']}% GPU"
            self.note(f"  loaded: {m['name']} ({format_size(m['size'])}, {where})")
        tips = system.speed_tips(snap)
        if tips:
            self.console.print(Text("Speed tips", style="bold"))
            for tip in tips:
                self.console.print(Text(f"  • {tip}"))
        else:
            self.note("No speed problems found.", style="green")

    def cmd_file(self, rest: str) -> None:
        from .cli import load_source

        parts = rest.split()
        if not parts:
            raise ValueError("Usage: /file PATH [LINES]   e.g. /file app.py 10-40")
        # Allow paths with spaces: a trailing "10-40" or "12" is the line range.
        lines = None
        if len(parts) > 1 and parts[-1].replace("-", "").isdigit():
            lines = parts[-1]
            parts = parts[:-1]
        source = load_source(" ".join(parts).strip('"'), lines)
        self.pending_context = source.selection
        self.pending_label = f"{source.path.name if source.path else 'stdin'}:{source.start}-{source.end}"
        self.note(f"📎 Attached {source.label} lines {source.start}-{source.end}. "
                  "Now ask about it.", style="green")

    def cmd_paste(self, _rest: str) -> None:
        self.note("Paste your code, then type /end on its own line.")
        pasted = []
        while True:
            try:
                line = self.console.input("")
            except (EOFError, KeyboardInterrupt):
                break
            if line.strip() == "/end":
                break
            pasted.append(line)
        if not pasted:
            self.note("Nothing pasted.")
            return
        self.pending_context = "\n".join(pasted)
        self.pending_label = f"pasted {len(pasted)} lines"
        self.note(f"📎 Attached {len(pasted)} lines. Now ask about it.", style="green")

    def cmd_manage(self, rest: str) -> None:
        if self.manager is None:
            self.error("TURBO Manager isn't available.")
            return
        if not rest:
            raise ValueError("Tell the manager what to do, e.g. /manage it's too slow")
        self.console.print(Text("◆ TURBO Manager", style=f"bold {ACCENT}"))
        self.manager.run(rest, ui=self)
        self.console.print()
