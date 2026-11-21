"""Prompt text. Kept in one place so it is easy to tune."""

from __future__ import annotations

import re

SYSTEM_PROMPT = """You are TURBO, a precise and thoughtful coding assistant running locally on the user's computer.

Voice: natural, calm and slightly formal. Be warm but not chatty. No filler,
no flattery, no emoji.

How to answer:
- Lead with the answer. For anything non-trivial, then reason step by step in
  short numbered steps. Keep simple answers short.
- Be precise: exact names, exact commands, exact line references.
- State limitations up front. You are a small local model: you can't browse
  the internet, you only see files the user shares with you, and you can be
  wrong. If you are not sure, say so plainly instead of guessing.
- Never invent functions, files, libraries, APIs or command options.
- If you need code or details you were not given, say exactly what is missing.
- Use earlier messages in this conversation for follow-up questions.
- Use Markdown, and fenced code blocks with a language tag for any code.

When explaining code:
- Start with one or two sentences on what it does overall.
- Then walk through the important parts. Point out bugs, risks or edge cases you notice.

When fixing code:
- First explain the bug: what is wrong, where it is, and why it fails.
- Then make the smallest change that fixes it. Keep the user's names, style,
  formatting and comments. Do not refactor, rename, reformat or add features
  that were not asked for.
- If you find no bug, say so plainly and leave the code unchanged."""

LANGUAGE_ALIASES = {
    "python": "python", "py": "python",
    "javascript": "javascript", "js": "javascript",
    "typescript": "typescript", "ts": "typescript",
    "javascriptreact": "jsx", "typescriptreact": "tsx",
    "csharp": "csharp", "cs": "csharp",
    "cpp": "cpp", "c": "c", "java": "java", "go": "go", "rust": "rust",
    "html": "html", "css": "css", "json": "json", "sql": "sql",
    "shellscript": "bash", "sh": "bash", "powershell": "powershell", "ps1": "powershell",
}

EXTENSION_LANGUAGES = {
    ".py": "python", ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".ts": "typescript", ".tsx": "tsx", ".jsx": "jsx", ".cs": "csharp",
    ".cpp": "cpp", ".cc": "cpp", ".h": "c", ".c": "c", ".java": "java",
    ".go": "go", ".rs": "rust", ".html": "html", ".css": "css", ".json": "json",
    ".sql": "sql", ".sh": "bash", ".ps1": "powershell",
}


def guess_language(filename: str | None, language: str | None) -> str:
    if language:
        return LANGUAGE_ALIASES.get(language.lower(), language.lower())
    if filename:
        for ext, lang in EXTENSION_LANGUAGES.items():
            if filename.lower().endswith(ext):
                return lang
    return ""


def fence(code: str, language: str = "") -> str:
    """Wrap code in a fence longer than any backtick run inside it."""
    longest = max((len(run) for run in re.findall(r"`+", code)), default=0)
    ticks = "`" * max(3, longest + 1)
    return f"{ticks}{language}\n{code}\n{ticks}"


def _source_label(filename: str | None, language: str) -> str:
    parts = []
    if filename:
        parts.append(f"`{filename}`")
    if language:
        parts.append(f"({language})")
    return " ".join(parts) or "the user's project"


def _context_block(context: str | None, language: str) -> str:
    if not context:
        return ""
    return (
        "\n\nSurrounding code, for reference only (do not explain or return it):\n"
        + fence(context, language)
    )


def ask_message(question: str, context: str | None = None, language: str = "") -> str:
    if not context:
        return question
    return f"{question}\n\nRelevant code:\n{fence(context, language)}"


def explain_message(
    code: str,
    filename: str | None,
    language: str,
    question: str | None = None,
    context: str | None = None,
) -> str:
    request = question.strip() if question else "Explain what this code does."
    return (
        f"{request}\n\nCode from {_source_label(filename, language)}:\n"
        f"{fence(code, language)}"
        f"{_context_block(context, language)}"
    )


def fix_message(
    code: str,
    filename: str | None,
    language: str,
    instruction: str | None = None,
    context: str | None = None,
) -> str:
    task = instruction.strip() if instruction else "Find and fix the bug in this code."
    return f"""{task}

Code from {_source_label(filename, language)}:
{fence(code, language)}{_context_block(context, language)}

Reply in exactly this format:

EXPLANATION:
<2-5 sentences: what the bug is, where, why it fails, and what you changed>

FIXED CODE:
```{language}
<the complete corrected version of the code above - every line, not only the changed ones>
```

Make the smallest possible change. If there is no bug, say so in EXPLANATION
and return the code unchanged."""
