"""TURBO command line: ask questions, explain and fix code from your terminal.

Examples:
    turbo ask "What does git rebase do?"
    turbo chat
    turbo explain app.py --lines 10-40
    turbo fix app.py --lines 10-40          # prints a diff
    turbo fix app.py --lines 10-40 --apply  # asks, then writes it
    turbo serve                             # HTTP server for VS Code
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from turbo_core.assistant import Assistant, InputError
from turbo_core.config import Settings
from turbo_core.llm import LLMError

CONTEXT_LINES = 40

# ANSI colors (Windows 10+ terminals support these).
RED, GREEN, CYAN, DIM, RESET = "\033[31m", "\033[32m", "\033[36m", "\033[2m", "\033[0m"


@dataclass
class Source:
    path: Path | None
    text: str           # the whole file
    start: int          # 1-based first selected line
    end: int            # 1-based last selected line
    selection: str
    context: str | None

    @property
    def label(self) -> str:
        return str(self.path) if self.path else "stdin"

    @property
    def display_name(self) -> str:
        """Path relative to the current folder when possible, for diff headers."""
        if self.path is None:
            return "stdin"
        try:
            return str(self.path.resolve().relative_to(Path.cwd().resolve()))
        except ValueError:
            return self.path.name


def parse_lines(spec: str | None, total: int) -> tuple[int, int]:
    if not spec:
        return 1, max(total, 1)
    try:
        if "-" in spec:
            start_text, end_text = spec.split("-", 1)
            start, end = int(start_text), int(end_text)
        else:
            start = end = int(spec)
    except ValueError:
        raise InputError(f"--lines must look like 12 or 10-40, got {spec!r}") from None
    if start < 1 or end < start or start > total:
        raise InputError(f"--lines {spec} is outside the file (1-{total}).")
    return start, min(end, total)


def load_source(file_arg: str, lines_spec: str | None) -> Source:
    if file_arg == "-":
        text = sys.stdin.read()
        path = None
    else:
        path = Path(file_arg)
        if not path.is_file():
            raise InputError(f"File not found: {file_arg}")
        try:
            # newline="" keeps CRLF so diffs and --apply preserve line endings.
            with path.open(encoding="utf-8", newline="") as handle:
                text = handle.read()
        except UnicodeDecodeError:
            raise InputError(f"{file_arg} is not a UTF-8 text file.") from None

    lines = text.splitlines(keepends=True)
    start, end = parse_lines(lines_spec, len(lines))
    selection = "".join(lines[start - 1:end])

    context = None
    if lines_spec:
        before = lines[max(0, start - 1 - CONTEXT_LINES):start - 1]
        after = lines[end:end + CONTEXT_LINES]
        if before or after:
            context = "".join(before) + "# >>> selected code goes here <<<\n" + "".join(after)
    return Source(path, text, start, end, selection, context)


def colorize_diff(diff: str, use_color: bool) -> str:
    if not use_color:
        return diff
    out = []
    for line in diff.splitlines():
        if line.startswith(("+++", "---")):
            out.append(f"{DIM}{line}{RESET}")
        elif line.startswith("+"):
            out.append(f"{GREEN}{line}{RESET}")
        elif line.startswith("-"):
            out.append(f"{RED}{line}{RESET}")
        elif line.startswith("@@"):
            out.append(f"{CYAN}{line}{RESET}")
        else:
            out.append(line)
    return "\n".join(out) + "\n"


def status(message: str) -> None:
    print(f"{DIM}{message}{RESET}" if sys.stderr.isatty() else message, file=sys.stderr)


class StreamPrinter:
    """Prints the answer as it arrives. Shows a 'thinking...' line until the
    first words come in, then erases it (on a real terminal)."""

    def __init__(self, waiting_message: str):
        self.started = False
        self.at_line_start = True
        self.tty = sys.stderr.isatty()
        if self.tty:
            sys.stderr.write(f"{DIM}{waiting_message}{RESET}")
            sys.stderr.flush()
        else:
            status(waiting_message)

    def __call__(self, text: str) -> None:
        if not self.started:
            text = text.lstrip()
            if not text:
                return
            self.started = True
            if self.tty:
                sys.stderr.write("\r\033[K")  # erase the "thinking..." line
                sys.stderr.flush()
        sys.stdout.write(text)
        sys.stdout.flush()
        self.at_line_start = text.endswith("\n")

    def finish(self) -> None:
        if self.tty and not self.started:
            sys.stderr.write("\r\033[K")
            sys.stderr.flush()
        if self.started and not self.at_line_start:
            sys.stdout.write("\n")
            sys.stdout.flush()


class FixExplanationFilter:
    """Passes through the explanation part of a fix reply and hides the rest.

    The reply looks like 'EXPLANATION: ... FIXED CODE: ```...```'. The code is
    shown afterwards as a diff, so streaming it too would just repeat it.
    """

    START = "EXPLANATION:"
    STOP = "FIXED CODE"

    def __init__(self, emit):
        self.emit = emit
        self.buffer = ""
        self.started = False   # label handled
        self.emitted = False   # some real text already sent
        self.hidden = False    # reached the code; drop the rest

    def __call__(self, text: str) -> None:
        if self.hidden:
            return
        self.buffer += text
        if not self.started:
            stripped = self.buffer.lstrip()
            if len(stripped) < len(self.START) and self.START.startswith(stripped.upper()):
                return  # might still be the label; wait
            if stripped.upper().startswith(self.START):
                stripped = stripped[len(self.START):]
            self.buffer = stripped
            self.started = True
        if not self.emitted:
            self.buffer = self.buffer.lstrip()

        index = self.buffer.upper().find(self.STOP)
        if index >= 0:
            self._send(self.buffer[:index].rstrip())
            self.buffer = ""
            self.hidden = True
            return
        # Hold back a tail that could be the start of the STOP marker, plus any
        # trailing whitespace (it may be the blank lines right before it).
        safe = len(self.buffer) - (len(self.STOP) - 1)
        if safe > 0:
            ready = self.buffer[:safe].rstrip()
            self._send(ready)
            self.buffer = self.buffer[len(ready):]

    def _send(self, text: str) -> None:
        if text:
            self.emit(text)
            self.emitted = True

    def finish(self) -> None:
        if not self.hidden:
            self._send(self.buffer.strip() if not self.emitted else self.buffer.rstrip())
        self.buffer = ""


def apply_fix(source: Source, fixed_code: str) -> None:
    lines = source.text.splitlines(keepends=True)
    with source.path.open(encoding="utf-8", newline="") as handle:
        if handle.read() != source.text:
            raise InputError(f"{source.path} changed while TURBO was working. Not applied.")
    new_text = "".join(lines[:source.start - 1]) + fixed_code + "".join(lines[source.end:])
    with source.path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(new_text)


# ---- commands ----------------------------------------------------------------

def fancy_terminal(args) -> bool:
    """Use the rich UI only on a real interactive terminal."""
    return (
        not getattr(args, "plain", False)
        and sys.stdin.isatty()
        and sys.stdout.isatty()
    )


def run_streamed(call, waiting_message: str, stream: bool, fancy: bool = False) -> str:
    """Run an assistant call, streaming its answer to the terminal if enabled.
    Returns the full answer; prints it at the end when not streaming."""
    if stream and fancy:
        from turbo_core.ui import stream_call
        from rich.console import Console

        result, _ = stream_call(Console(highlight=False), call, waiting_message.rstrip("."))
        return result
    if not stream:
        status(waiting_message)
        answer = call(None)
        print(answer)
        return answer
    printer = StreamPrinter(waiting_message)
    try:
        return call(printer)
    finally:
        printer.finish()


def cmd_ask(assistant: Assistant, args) -> int:
    context = None
    if args.file:
        context = load_source(args.file, args.lines).selection
    question = " ".join(args.question)
    run_streamed(
        lambda on_token: assistant.ask(question, session_id=args.session, context=context, on_token=on_token),
        "TURBO is thinking...",
        not args.no_stream,
        fancy_terminal(args),
    )
    return 0


def cmd_explain(assistant: Assistant, args) -> int:
    source = load_source(args.file, args.lines)
    run_streamed(
        lambda on_token: assistant.explain(
            source.selection,
            filename=source.path.name if source.path else None,
            question=args.question,
            context=source.context,
            session_id=args.session,
            on_token=on_token,
        ),
        f"Explaining {source.label} lines {source.start}-{source.end}...",
        not args.no_stream,
        fancy_terminal(args),
    )
    return 0


def cmd_fix(assistant: Assistant, args) -> int:
    source = load_source(args.file, args.lines)
    waiting = f"Fixing {source.label} lines {source.start}-{source.end}..."
    fix_kwargs = dict(
        filename=source.display_name,
        instruction=args.message,
        context=source.context,
        session_id=args.session,
        start_line=source.start,
    )

    if args.no_stream:
        status(waiting)
        result = assistant.fix(source.selection, **fix_kwargs)
        print(result.explanation, end="\n\n")
    else:
        printer = StreamPrinter(waiting)
        explanation = FixExplanationFilter(printer)
        try:
            result = assistant.fix(source.selection, on_token=explanation, **fix_kwargs)
            explanation.finish()
        finally:
            printer.finish()
        if not printer.started:  # nothing streamed (unexpected format)
            print(result.explanation)
        print()
    for warning in result.warnings:
        print(f"Warning: {warning}", file=sys.stderr)
    if not result.changed:
        print("No changes suggested.")
        return 0

    print(colorize_diff(result.diff, sys.stdout.isatty() and not args.no_color), end="")

    if args.apply:
        if source.path is None:
            raise InputError("--apply needs a file, not stdin.")
        if not args.yes:
            answer = input(f"\nApply this change to {source.path}? [y/N] ").strip().lower()
            if answer not in ("y", "yes"):
                print("Not applied.")
                return 0
        apply_fix(source, result.fixed_code)
        print(f"Applied to {source.path}.")
    return 0


def cmd_chat(assistant: Assistant, args) -> int:
    if fancy_terminal(args) and not args.no_stream:
        from turbo_core.ui import ChatUI

        return ChatUI(assistant, args.session, manager=make_manager(assistant)).run()
    return plain_chat(assistant, args)


def make_manager(assistant: Assistant):
    """TURBO Manager for /manage, or None if it can't be loaded."""
    try:
        from turbo_core.manager import Manager
    except ImportError:
        return None
    return Manager(assistant)


def plain_chat(assistant: Assistant, args) -> int:
    print("TURBO chat - local coding assistant. Ask anything; follow-ups remember the conversation.")
    print("Commands: /file PATH [LINES]  /paste  /reset  /exit     (turbo --help for more)")
    pending_context: str | None = None
    while True:
        try:
            line = input(f"\n{CYAN}you>{RESET} " if sys.stdout.isatty() else "you> ")
            line = line.lstrip("﻿").strip()  # PowerShell pipes add a BOM
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        if line in ("/exit", "/quit"):
            return 0
        if line == "/reset":
            assistant.reset(args.session)
            pending_context = None
            print("Conversation cleared.")
            continue
        if line.startswith("/file "):
            parts = line[len("/file "):].split()
            try:
                source = load_source(parts[0], parts[1] if len(parts) > 1 else None)
            except InputError as error:
                print(error)
                continue
            pending_context = source.selection
            print(f"Attached {source.label} lines {source.start}-{source.end}. Now ask about it.")
            continue
        if line == "/paste":
            print("Paste code, then type /end on its own line.")
            pasted = []
            while True:
                try:
                    pasted_line = input()
                except EOFError:
                    break
                if pasted_line.strip() == "/end":
                    break
                pasted.append(pasted_line)
            pending_context = "\n".join(pasted)
            print(f"Attached {len(pasted)} lines. Now ask about it.")
            continue

        try:
            print()
            run_streamed(
                lambda on_token: assistant.ask(
                    line, session_id=args.session, context=pending_context, on_token=on_token
                ),
                "thinking...",
                not args.no_stream,
            )
            pending_context = None
        except (LLMError, InputError) as error:
            print(f"Error: {error}")
        except KeyboardInterrupt:
            print("\n(stopped)")


def cmd_reset(assistant: Assistant, args) -> int:
    removed = assistant.reset(args.session)
    print(f"Cleared session '{args.session}' ({removed} messages).")
    return 0


def model_installed(name: str, installed: list[str]) -> bool:
    return name in installed or (":" not in name and f"{name}:latest" in installed)


def check_port(host: str, port: int) -> tuple[str, str]:
    """('free'|'turbo'|'busy', detail) for the server port."""
    import json
    import socket
    import urllib.request

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        if sock.connect_ex((host, port)) != 0:
            return "free", ""
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/health", timeout=3) as response:
            if json.loads(response.read()).get("ok"):
                return "turbo", ""
    except Exception as error:  # noqa: BLE001 - anything else on the port
        return "busy", str(error)
    return "busy", ""


def cmd_doctor(assistant: Assistant, args) -> int:
    from turbo_core import __version__

    settings = assistant.settings
    color = sys.stdout.isatty()
    ok_mark = f"{GREEN}✓{RESET}" if color else "OK  "
    bad_mark = f"{RED}✗{RESET}" if color else "FAIL"
    note_mark = f"{CYAN}i{RESET}" if color else "    "
    problems = 0

    def report(good: bool | None, text: str, fix: str = "") -> None:
        nonlocal problems
        mark = note_mark if good is None else ok_mark if good else bad_mark
        print(f" {mark} {text}")
        if fix:
            print(f"      -> {fix}")
        if good is False:
            problems += 1

    print(f"TURBO {__version__} doctor\n")
    report(None, f"Python {sys.version.split()[0]}  ({sys.executable})")
    report(None, f"Data folder: {settings.db_path.parent}")

    try:
        installed = assistant.llm.list_models()
    except LLMError:
        report(False, f"Ollama is not reachable at {settings.ollama_url}",
               "Install it from https://ollama.com, then start the Ollama app (or run: ollama serve)")
        installed = None
    else:
        report(True, f"Ollama is running at {settings.ollama_url} ({len(installed)} models installed)")

    for role, name in (("chat", settings.chat_model), ("code", settings.code_model)):
        if installed is None:
            report(None, f"{role} model {name}: can't check until Ollama is running")
        elif model_installed(name, installed):
            report(True, f"{role} model {name} is installed")
        else:
            report(False, f"{role} model {name} is not installed", f"ollama pull {name}")

    state, detail = check_port(settings.host, settings.port)
    if state == "free":
        report(True, f"Port {settings.port} is free for `turbo serve` (VS Code extension)")
    elif state == "turbo":
        report(True, f"A TURBO server is already running on port {settings.port}")
    else:
        report(False, f"Port {settings.port} is used by another program",
               "Set TURBO_PORT to another number (and turbo.serverUrl in VS Code)")

    print()
    if problems:
        print(f"{problems} problem(s) found. Fix the items marked above, then run `turbo doctor` again.")
        return 1
    print("All good. Try: turbo")
    return 0


def cmd_serve(_assistant: Assistant, args) -> int:
    from turbo_core.server import serve

    serve(Settings.from_env())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="turbo", description="TURBO local coding assistant")
    parser.add_argument("-s", "--session", default="cli", help="conversation name (default: cli)")
    parser.add_argument(
        "--no-stream", action="store_true", help="print answers all at once instead of word by word"
    )
    parser.add_argument(
        "--plain", action="store_true", help="simple text output (no colors, banner or live rendering)"
    )
    parser.set_defaults(func=cmd_chat)  # plain `turbo` opens the chat
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("ask", help="ask a question")
    p.add_argument("question", nargs="+")
    p.add_argument("-f", "--file", help="attach a file as context")
    p.add_argument("-l", "--lines", help="only these lines of --file, e.g. 10-40")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("chat", help="interactive chat with follow-ups")
    p.set_defaults(func=cmd_chat)

    p = sub.add_parser("explain", help="explain a file or some lines of it")
    p.add_argument("file", help="path, or - for stdin")
    p.add_argument("-l", "--lines", help="e.g. 10-40")
    p.add_argument("-q", "--question", help="specific question about the code")
    p.set_defaults(func=cmd_explain)

    p = sub.add_parser("fix", help="find and fix a bug; prints a unified diff")
    p.add_argument("file", help="path, or - for stdin")
    p.add_argument("-l", "--lines", help="e.g. 10-40")
    p.add_argument("-m", "--message", help="what to fix, e.g. 'crashes on empty list'")
    p.add_argument("--apply", action="store_true", help="write the fix to the file (asks first)")
    p.add_argument("-y", "--yes", action="store_true", help="with --apply, don't ask")
    p.add_argument("--no-color", action="store_true")
    p.set_defaults(func=cmd_fix)

    p = sub.add_parser("reset", help="forget the conversation for --session")
    p.set_defaults(func=cmd_reset)

    p = sub.add_parser("doctor", help="check that Ollama, the models and the server port are ready")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("serve", help="run the local HTTP server for the VS Code extension")
    p.set_defaults(func=cmd_serve)
    return parser


def main(argv: list[str] | None = None, assistant: Assistant | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    args = build_parser().parse_args(argv)
    try:
        if args.command == "serve":
            return cmd_serve(None, args)
        assistant = assistant or Assistant.from_settings()
        return args.func(assistant, args)
    except InputError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    except LLMError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except ValueError as error:  # bad TURBO_* setting
        print(f"Config error: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nCancelled.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
