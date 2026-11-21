"""Interactive terminal UI for `turbo` (rich + prompt_toolkit).

Only used on a real terminal. Pipes and tests use the plain loop in cli.py.
"""

from __future__ import annotations

import html
import queue
import threading
import time
from pathlib import Path
from typing import Any, Callable

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory, InMemoryHistory
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
    "/build": "build a project on disk: /build NAME what to build  (continue: /build NAME more)",
    "/new": "start a fresh project: /new NAME what to build",
    "/auto": "toggle hands-free building (no [Y/n] before commands or overwrites)",
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
        self.show_elapsed = True

    def __rich_console__(self, console, options):
        elapsed = time.monotonic() - self.start
        if not self.text.strip():
            suffix = f"... {elapsed:.0f}s" if self.show_elapsed else ""
            self.spinner.update(text=Text(f" {self.label}{suffix}", style=DIM))
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
        self.builder = None   # active build-mode project
        self.auto = False     # /auto: build without confirmations
        self.building = False  # a build is running right now

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
        where = f"{snap.backend_label} (cloud, free tier)" if snap.is_cloud else "Ollama"
        if snap.ollama_ok:
            line.append("● ", style="green")
            line.append(f"{where} connected" if snap.is_cloud else "Ollama running", style="green")
        else:
            line.append("● ", style="red")
            line.append(f"{where} not reachable", style="red")
        line.append("   chat ", style=DIM)
        line.append(snap.chat_model, style="cyan")
        line.append("   code ", style=DIM)
        line.append(snap.code_model, style="cyan")
        if snap.ram_free_gb is not None and not snap.is_cloud:
            line.append(f"   RAM {snap.ram_free_gb} / {snap.ram_total_gb} GB free", style=DIM)
        self.console.print(line)

        warnings = []
        if snap.ollama_ok:
            for role, name in (("chat", snap.chat_model), ("code", snap.code_model)):
                if not snap.is_installed(name):
                    fix = ("run `turbo setup` to pick models" if snap.is_cloud
                           else f"or run `ollama pull {name}`")
                    warnings.append(f"{role} model {name} isn't available: /model to switch, {fix}")
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
        building = ""
        if self.builder is not None:
            auto = " auto" if self.auto else ""
            building = (f" · <style fg='ansigreen'>build: {html.escape(self.builder_label())}{auto}</style>")
        return HTML(
            f" chat <style fg='ansicyan'>{html.escape(settings.chat_model)}</style>"
            f" · code <style fg='ansicyan'>{html.escape(settings.code_model)}</style>{attached}{building}"
            " · /help · Ctrl+C stop · Ctrl+L clear"
        )

    def error(self, error: Exception | str) -> None:
        self.console.print(Text(f"✗ {error}", style="red"))

    def note(self, text: str, style: str = DIM) -> None:
        self.console.print(Text(text, style=style))

    def markdown(self, text: str) -> None:
        self.console.print(Markdown(text))

    def show_diff(self, diff: str) -> None:
        from rich.syntax import Syntax

        self.console.print(Syntax(diff or "(no changes)", "diff", theme="ansi_dark", background_color="default"))

    def show_output(self, text: str) -> None:
        if text.strip():
            self.console.print(Text(text.rstrip(), style=DIM))

    def wait(self, func: Callable[[], Any], label: str = "Working") -> Any:
        """Run func in a worker thread with a spinner; Ctrl+C raises KeyboardInterrupt
        right away instead of waiting for func to finish."""
        done: queue.Queue = queue.Queue()

        def worker() -> None:
            try:
                done.put((True, func()))
            except BaseException as error:  # noqa: BLE001 - handed to the main thread
                done.put((False, error))

        threading.Thread(target=worker, daemon=True).start()
        view = _StreamView(label)
        with Live(view, console=self.console, refresh_per_second=12, transient=True):
            while True:
                try:
                    ok, value = done.get(timeout=0.1)
                except queue.Empty:
                    continue
                if ok:
                    return value
                raise value

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
            # Typed messages (Up arrow) are only remembered for this session,
            # unless the user opted in to saving history.
            history=FileHistory(str(home / "history")) if self.assistant.settings.save_history
            else InMemoryHistory(),
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
            if self.offer_build(line):
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
            result, _ = self.with_choices(lambda: stream_call(
                self.console,
                lambda on_token: self.assistant.ask(
                    question, session_id=self.session_id, context=context, on_token=on_token
                ),
            ))
        except (LLMError, InputError) as error:
            self.error(error)
            return
        if result is not None:
            self.pending_context = None
            self.pending_label = ""
            self.route_note()
            if self.assistant.last_reply_truncated:
                self.note(f"(answer cut off at {self.assistant.settings.max_tokens} tokens: "
                          "ask \"continue\", or raise TURBO_MAX_TOKENS)")
        self.console.print()

    # -- free-limit fallback

    def with_choices(self, func: Callable[[], Any]) -> Any:
        """Run func; if a free limit is reached, ask what to do and retry."""
        llm = self.assistant.llm
        if not hasattr(llm, "choose"):
            return func()
        from .fallback import run_with_choices

        return run_with_choices(func, llm, self.choose, self.countdown)

    def choose(self, decision) -> Any:
        """Show the options for a reached limit and return the user's pick."""
        from .fallback import format_seconds

        if self.auto and self.building and decision.kind == "limit":
            # Hands-free build: keep going on the next free cloud model
            # (never local Ollama automatically, since it uses RAM).
            cloud = next((o for o in decision.options if o.kind == "switch" and o.backend == "cloud"), None)
            if cloud is not None:
                self.note(f"  ⚠ {decision.message} Auto: continuing with {cloud.model}.", style="yellow")
                return cloud
        self.console.print(Text(f"⚠ {decision.message}", style="yellow"))
        if decision.advice:
            self.console.print(Text(f"  {decision.advice}", style="yellow"))
        if [o.kind for o in decision.options] == ["cancel"]:
            self.note("  No other free model can take this request right now.")
            return None
        width = max(len(o.label) for o in decision.options)
        for number, option in enumerate(decision.options, 1):
            when = "" if option.kind == "cancel" else f"~{format_seconds(option.estimate_seconds)} to finish"
            self.console.print(Text.assemble((f"  {number}. ", "cyan"), (option.label.ljust(width + 3), ""),
                                             (when, DIM)))
        while True:
            try:
                answer = self.console.input(f"[yellow]Continue? Choose 1-{len(decision.options)} [1]:[/yellow] ")
            except (KeyboardInterrupt, EOFError):
                self.console.print()
                return None
            answer = answer.strip() or "1"
            if answer.isdigit() and 1 <= int(answer) <= len(decision.options):
                option = decision.options[int(answer) - 1]
                return None if option.kind == "cancel" else option
            self.note(f"Please type a number from 1 to {len(decision.options)}.")

    def countdown(self, seconds: float, model: str = "") -> None:
        """Wait with a visible countdown. Ctrl+C cancels."""
        from .fallback import RequestCancelled

        end = time.monotonic() + seconds
        view = _StreamView(f"Waiting for {model or 'the free limit'} to reset")
        view.show_elapsed = False
        try:
            with Live(view, console=self.console, refresh_per_second=8, transient=True):
                while time.monotonic() < end:
                    view.label = (f"Waiting for {model or 'the free limit'} to reset: "
                                  f"{end - time.monotonic():.0f}s left")
                    time.sleep(0.1)
        except KeyboardInterrupt:
            raise RequestCancelled("Cancelled.") from None

    def route_note(self) -> None:
        note = getattr(self.assistant.llm, "last_route_note", None)
        if note:
            self.note(f"({note})")

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
            "/build": self.cmd_build,
            "/new": lambda rest: self.cmd_build(rest, fresh=True),
            "/auto": self.cmd_auto,
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
        if self.assistant.settings.is_cloud:
            raise ValueError(f"{name} isn't one of the free models on this provider. See /models.")
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
            raise LLMError(snap.backend_error or "Ollama isn't running. Start the Ollama app (or run `ollama serve`).")
        if snap.is_cloud:
            table = Table(box=None, padding=(0, 2), header_style=DIM)
            table.add_column(f"free models on {snap.backend_label}")
            table.add_column("in use as")
            for m in snap.installed:
                roles = [r for r, n in (("chat", snap.chat_model), ("code", snap.code_model)) if n == m["name"]]
                table.add_row(m["name"], ", ".join(roles), style="cyan" if roles else None)
            self.console.print(table)
            self.note("Switch with /model NAME. Free limits are counted per model.")
            return
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
        self.print_limits()
        if snap.is_cloud:
            return
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

    def print_limits(self) -> None:
        """Free-tier usage left per model (cloud only), from the provider's headers."""
        report = getattr(self.assistant.llm, "limits_report", None)
        if report is None:
            return
        from .fallback import format_seconds

        rows = report()
        if not rows:
            self.note("  Free limits: shown here after your first question.")
            return
        table = Table(box=None, padding=(0, 2), header_style=DIM)
        for column in ("model", "requests left today", "tokens left this minute", "status"):
            table.add_column(column)
        for row in rows:
            def left(remaining, limit):
                return "?" if remaining is None else f"{remaining:,} / {limit:,}" if limit else f"{remaining:,}"
            cooling = row.get("cooling_seconds", 0)
            state = f"at limit, resets in {format_seconds(cooling)}" if cooling > 0 else "ok"
            table.add_row(row["model"], left(row.get("remaining_requests"), row.get("limit_requests")),
                          left(row.get("remaining_tokens"), row.get("limit_tokens")), state,
                          style="yellow" if cooling > 0 else None)
        self.console.print(table)

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

    # -- build mode

    def builder_label(self) -> str:
        if self.builder is None:
            return ""
        try:
            return "./" + self.builder.workspace.relative_to(Path.cwd().resolve()).as_posix()
        except ValueError:
            return str(self.builder.workspace)

    def file_event(self, action: str, rel: str, lines: int) -> None:
        mark, style = ("✔", "green") if action == "Created" else ("✎", "cyan")
        self.console.print(Text(f"  {mark} {action} {rel} ({lines} lines)", style=style))

    def cmd_auto(self, rest: str) -> None:
        arg = rest.strip().lower()
        self.auto = {"on": True, "off": False}.get(arg, not self.auto)
        if self.builder is not None:
            self.builder.auto = self.auto
        if self.auto:
            self.note("Auto mode ON: builds run commands and overwrite files without asking. "
                      "Files stay inside the project folder; dangerous commands are still refused. "
                      "/auto again to turn it off.", style="yellow")
        else:
            self.note("Auto mode OFF: TURBO asks [Y/n] before running commands or overwriting files.")

    def cmd_build(self, rest: str, fresh: bool = False) -> None:
        name, _, request = rest.strip().partition(" ")
        if not name:
            if self.builder is None:
                raise ValueError("Usage: /build NAME what to build   e.g. /build todo-app a Flask to-do list")
            name = str(self.builder.workspace)
        elif self.builder is not None and not fresh:
            target = (Path.cwd() / name).resolve()
            if target != self.builder.workspace and not target.exists():
                # "/build add a delete button" probably means: keep going on this project.
                try:
                    answer = self.console.input(
                        f"[yellow]? Continue {self.builder_label()} with \"{rest.strip()}\", or start a new "
                        f"project in ./{name}? [C/n][/yellow] ").strip().lower()
                except (KeyboardInterrupt, EOFError):
                    self.console.print()
                    return
                if answer in ("", "c", "continue", "y", "yes"):
                    name, request = str(self.builder.workspace), rest.strip()
        self.start_build(name, request.strip(), fresh=fresh)

    def offer_build(self, line: str) -> bool:
        """For a plain request like 'create a Flask app in ./my-app', offer build mode."""
        from .builder import looks_like_build_request, suggest_folder

        if not looks_like_build_request(line):
            return False
        folder = suggest_folder(line)
        try:
            answer = self.console.input(
                f"[yellow]? Build this as a project on disk? Folder [./{folder}] "
                "(Enter = yes, n = just chat, or type a folder name):[/yellow] ").strip()
        except (KeyboardInterrupt, EOFError):
            self.console.print()
            return True
        if answer.lower() in ("n", "no"):
            return False
        self.start_build(answer if answer and answer.lower() not in ("y", "yes") else folder, line)
        return True

    def start_build(self, folder: str, request: str, fresh: bool = False) -> None:
        from .builder import Builder

        workspace = Path(folder.strip().strip('"')).expanduser()
        workspace = (workspace if workspace.is_absolute() else Path.cwd() / workspace).resolve()
        if workspace in (Path.home().resolve(), Path(workspace.anchor)):
            raise ValueError("Pick a project folder, not your home folder or a whole drive.")
        same = self.builder is not None and self.builder.workspace == workspace
        if not same or fresh:
            if workspace.is_dir() and any(workspace.iterdir()):
                count = sum(1 for _ in workspace.rglob("*"))
                self.note(f"{workspace} already exists ({count} items). TURBO will work with what's there "
                          "and ask before overwriting files.")
            self.builder = Builder(self.assistant, workspace, ui=self, auto=self.auto)
        if not request:
            try:
                request = self.console.input(f"[cyan]What should I build in {self.builder_label()}?[/cyan] ").strip()
            except (KeyboardInterrupt, EOFError):
                self.console.print()
                return
            if not request:
                return
        self.console.print(Text(f"◆ TURBO build · {self.builder_label()}" + (" · auto" if self.auto else ""),
                                style=f"bold {ACCENT}"))
        self.building = True
        try:
            answer = self.builder.run(request)
        except KeyboardInterrupt:
            self.note("(build stopped - /build again to continue)")
            return
        except (LLMError, InputError) as error:
            self.error(error)
            return
        finally:
            self.building = False
        self.markdown(answer)
        created, updated = self.builder.created, self.builder.updated
        self.note(f"{len(created)} files created, {len(updated)} updated in {self.builder.workspace}. "
                  f"Continue with: /build {self.builder_label()} <what to change>", style="green")
        self.console.print()

    def cmd_manage(self, rest: str) -> None:
        if self.manager is None:
            self.error("TURBO Manager isn't available.")
            return
        if not rest:
            raise ValueError("Tell the manager what to do, e.g. /manage it's too slow")
        self.console.print(Text("◆ TURBO Manager", style=f"bold {ACCENT}"))
        self.manager.run(rest, ui=self)
        self.console.print()
