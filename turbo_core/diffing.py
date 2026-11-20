"""Turn a model's corrected code into a clean, always-valid unified diff.

Small local models are bad at writing diffs by hand, so the model returns the
corrected code and we compute the diff ourselves with difflib.
"""

from __future__ import annotations

import difflib
import os
import re
from dataclasses import dataclass

_FENCE_RE = re.compile(
    r"^(?P<ticks>`{3,})[^\n`]*\n(?P<body>.*?)^(?P=ticks)[ \t]*$",
    re.DOTALL | re.MULTILINE,
)
_HUNK_RE = re.compile(r"^@@ -(\d+)(,\d+)? \+(\d+)(,\d+)? @@", re.MULTILINE)


@dataclass
class ParsedFix:
    explanation: str
    code: str | None


def parse_fix_reply(reply: str) -> ParsedFix:
    """Split a reply into its explanation and the corrected code block."""
    marker = re.search(r"FIXED CODE\s*:?", reply, re.IGNORECASE)
    search_from = marker.end() if marker else 0

    blocks = list(_FENCE_RE.finditer(reply, search_from))
    if not blocks and marker:
        blocks = list(_FENCE_RE.finditer(reply))
    if not blocks:
        return ParsedFix(explanation=reply.strip(), code=None)

    # After the marker, the first block is the answer. Without a marker, the
    # last block is the most likely final version.
    block = blocks[0] if marker else blocks[-1]
    code = block.group("body")
    if code.endswith("\n"):
        code = code[:-1]

    explanation_end = marker.start() if marker else block.start()
    explanation = reply[:explanation_end]
    explanation = re.sub(r"^\s*EXPLANATION\s*:?\s*", "", explanation, flags=re.IGNORECASE)
    return ParsedFix(explanation=explanation.strip(), code=code)


@dataclass
class CodeShape:
    """Formatting details of the original code, restored after the fix."""

    eol: str
    trailing_newline: bool
    margin: str


def analyze(code: str) -> CodeShape:
    eol = "\r\n" if "\r\n" in code else "\n"
    lf = code.replace("\r\n", "\n")
    leading = [
        line[: len(line) - len(line.lstrip())]
        for line in lf.split("\n")
        if line.strip()
    ]
    margin = os.path.commonprefix(leading) if leading else ""
    return CodeShape(eol=eol, trailing_newline=lf.endswith("\n"), margin=margin)


def dedent(code: str, shape: CodeShape) -> str:
    """Normalize to LF and strip the shared left margin (for the prompt)."""
    lf = code.replace("\r\n", "\n")
    if not shape.margin:
        return lf
    return "\n".join(
        line[len(shape.margin):] if line.startswith(shape.margin) else line
        for line in lf.split("\n")
    )


def restore(fixed: str, shape: CodeShape) -> str:
    """Re-apply the original margin, line endings and trailing newline."""
    lf = fixed.replace("\r\n", "\n").rstrip("\n")
    lines = lf.split("\n")

    if shape.margin:
        nonblank = [line for line in lines if line.strip()]
        already_indented = nonblank and all(line.startswith(shape.margin) for line in nonblank)
        if not already_indented:
            lines = [shape.margin + line if line.strip() else line for line in lines]

    result = "\n".join(lines)
    if shape.trailing_newline:
        result += "\n"
    return result.replace("\n", shape.eol) if shape.eol != "\n" else result


def unified_diff(original: str, fixed: str, filename: str = "code", start_line: int = 1) -> str:
    """Unified diff of two strings. start_line shifts hunk numbers for selections."""
    a = original.replace("\r\n", "\n").splitlines(keepends=True)
    b = fixed.replace("\r\n", "\n").splitlines(keepends=True)
    name = filename.replace("\\", "/")

    out = []
    for line in difflib.unified_diff(a, b, fromfile=f"a/{name}", tofile=f"b/{name}"):
        if not line.endswith("\n"):
            line += "\n\\ No newline at end of file\n"
        out.append(line)
    diff = "".join(out)

    if start_line != 1 and diff:
        offset = start_line - 1

        def shift(match: re.Match) -> str:
            old = int(match.group(1)) + offset if match.group(1) != "0" else 0
            new = int(match.group(3)) + offset if match.group(3) != "0" else 0
            return f"@@ -{old}{match.group(2) or ''} +{new}{match.group(4) or ''} @@"

        diff = _HUNK_RE.sub(shift, diff)
    return diff
