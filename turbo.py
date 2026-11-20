"""TURBO command line: ask questions, explain and fix code from your terminal.

Examples:
    python turbo.py ask "What does git rebase do?"
    python turbo.py chat
    python turbo.py explain app.py --lines 10-40
    python turbo.py fix app.py --lines 10-40          # prints a diff
    python turbo.py fix app.py --lines 10-40 --apply  # asks, then writes it
    python turbo.py serve                             # HTTP server for VS Code
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


def apply_fix(source: Source, fixed_code: str) -> None:
    lines = source.text.splitlines(keepends=True)
    with source.path.open(encoding="utf-8", newline="") as handle:
        if handle.read() != source.text:
            raise InputError(f"{source.path} changed while TURBO was working. Not applied.")
    new_text = "".join(lines[:source.start - 1]) + fixed_code + "".join(lines[source.end:])
    with source.path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(new_text)


# ---- commands ----------------------------------------------------------------

def cmd_ask(assistant: Assistant, args) -> int:
    context = None
    if args.file:
        context = load_source(args.file, args.lines).selection
    status("TURBO is thinking...")
    print(assistant.ask(" ".join(args.question), session_id=args.session, context=context))
    return 0


def cmd_explain(assistant: Assistant, args) -> int:
    source = load_source(args.file, args.lines)
    status(f"Explaining {source.label} lines {source.start}-{source.end}...")
    print(assistant.explain(
        source.selection,
        filename=source.path.name if source.path else None,
        question=args.question,
        context=source.context,
        session_id=args.session,
    ))
    return 0


def cmd_fix(assistant: Assistant, args) -> int:
    source = load_source(args.file, args.lines)
    status(f"Fixing {source.label} lines {source.start}-{source.end}...")
    result = assistant.fix(
        source.selection,
        filename=source.display_name,
        instruction=args.message,
        context=source.context,
        session_id=args.session,
        start_line=source.start,
    )

    print(result.explanation, end="\n\n")
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
    print("TURBO chat. Commands: /file PATH [LINES]  /paste  /reset  /exit")
    pending_context: str | None = None
    while True:
        try:
            line = input(f"\n{CYAN}you>{RESET} " if sys.stdout.isatty() else "you> ").strip()
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
            status("thinking...")
            answer = assistant.ask(line, session_id=args.session, context=pending_context)
            pending_context = None
            print(f"\n{answer}")
        except (LLMError, InputError) as error:
            print(f"Error: {error}")


def cmd_reset(assistant: Assistant, args) -> int:
    removed = assistant.reset(args.session)
    print(f"Cleared session '{args.session}' ({removed} messages).")
    return 0


def cmd_serve(_assistant: Assistant, args) -> int:
    from turbo_core.server import serve

    serve(Settings.from_env())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="turbo", description="TURBO local coding assistant")
    parser.add_argument("-s", "--session", default="cli", help="conversation name (default: cli)")
    sub = parser.add_subparsers(dest="command", required=True)

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
