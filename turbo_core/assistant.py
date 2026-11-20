"""The assistant: Q&A, explain code, fix code. Shared by the CLI and server."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from . import diffing, prompts
from .config import Settings
from .conversation import ConversationStore
from .llm import OllamaClient

MAX_CODE_CHARS = 60_000
MAX_CONTEXT_CHARS = 8_000


class ChatClient(Protocol):
    def chat(self, messages: list[dict[str, str]], model: str, temperature: float = ...) -> str: ...


class InputError(ValueError):
    """Bad request from the user (empty code, too large, ...)."""


@dataclass
class FixResult:
    explanation: str
    original_code: str
    fixed_code: str
    diff: str
    changed: bool
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "explanation": self.explanation,
            "fixed_code": self.fixed_code,
            "diff": self.diff,
            "changed": self.changed,
            "warnings": self.warnings,
        }


class Assistant:
    def __init__(self, llm: ChatClient, store: ConversationStore, settings: Settings):
        self.llm = llm
        self.store = store
        self.settings = settings

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "Assistant":
        settings = settings or Settings.from_env()
        llm = OllamaClient(
            base_url=settings.ollama_url,
            timeout=settings.timeout_seconds,
            context_tokens=settings.context_tokens,
        )
        return cls(llm, ConversationStore(settings.db_path), settings)

    # ---- public API -------------------------------------------------------

    def ask(
        self,
        question: str,
        session_id: str = "default",
        context: str | None = None,
        language: str | None = None,
    ) -> str:
        question = _require(question, "question")
        context = _limit(context, MAX_CODE_CHARS, "context") if context else None
        message = prompts.ask_message(question, context, prompts.guess_language(None, language))
        return self._converse(session_id, message, model=None)

    def explain(
        self,
        code: str,
        filename: str | None = None,
        language: str | None = None,
        question: str | None = None,
        context: str | None = None,
        session_id: str = "default",
    ) -> str:
        code = _limit(_require(code, "code"), MAX_CODE_CHARS, "code")
        lang = prompts.guess_language(filename, language)
        shape = diffing.analyze(code)
        message = prompts.explain_message(
            diffing.dedent(code, shape), filename, lang, question, _trim_context(context)
        )
        return self._converse(session_id, message, self.settings.code_model)

    def fix(
        self,
        code: str,
        filename: str | None = None,
        language: str | None = None,
        instruction: str | None = None,
        context: str | None = None,
        session_id: str = "default",
        start_line: int = 1,
    ) -> FixResult:
        code = _limit(_require(code, "code"), MAX_CODE_CHARS, "code")
        lang = prompts.guess_language(filename, language)
        shape = diffing.analyze(code)
        message = prompts.fix_message(
            diffing.dedent(code, shape), filename, lang, instruction, _trim_context(context)
        )
        reply = self._converse(session_id, message, self.settings.code_model, temperature=0.1)

        parsed = diffing.parse_fix_reply(reply)
        if parsed.code is None:
            return FixResult(
                explanation=parsed.explanation,
                original_code=code,
                fixed_code=code,
                diff="",
                changed=False,
                warnings=["The model did not return corrected code, so nothing was changed."],
            )

        fixed = diffing.restore(parsed.code, shape)
        diff = diffing.unified_diff(code, fixed, filename or "selection", start_line)
        warnings = []
        if _looks_truncated(code, fixed):
            warnings.append(
                "The fix is much shorter than the original. The model may have returned "
                "only part of the code - review the diff carefully."
            )
        return FixResult(
            explanation=parsed.explanation or "(no explanation given)",
            original_code=code,
            fixed_code=fixed,
            diff=diff,
            changed=bool(diff),
            warnings=warnings,
        )

    def reset(self, session_id: str = "default") -> int:
        return self.store.clear(session_id)

    # ---- internals --------------------------------------------------------

    def _converse(
        self, session_id: str, message: str, model: str | None, temperature: float = 0.2
    ) -> str:
        """Send message with session history. model=None picks automatically:
        the code model if code is in play, otherwise the faster chat model."""
        session_id = (session_id or "default").strip()[:100] or "default"
        history = self.store.history(session_id, self.settings.history_chars)
        messages = [{"role": "system", "content": prompts.SYSTEM_PROMPT}, *history]
        messages.append({"role": "user", "content": message})
        if model is None:
            has_code = any("```" in m["content"] for m in messages[1:])
            model = self.settings.code_model if has_code else self.settings.chat_model
        reply = self.llm.chat(messages, model=model, temperature=temperature)
        # Saved only after a successful reply, so failures don't pollute history.
        self.store.add_exchange(session_id, message, reply)
        return reply


def _require(value: str | None, name: str) -> str:
    if value is None or not str(value).strip():
        raise InputError(f"'{name}' is empty.")
    return str(value)


def _limit(value: str, max_chars: int, name: str) -> str:
    if len(value) > max_chars:
        raise InputError(
            f"'{name}' is {len(value):,} characters; the limit is {max_chars:,}. "
            "Select a smaller piece of code."
        )
    return value


def _trim_context(context: str | None) -> str | None:
    if not context or not context.strip():
        return None
    if len(context) <= MAX_CONTEXT_CHARS:
        return context
    half = MAX_CONTEXT_CHARS // 2
    return context[:half] + "\n# ... (middle of file omitted) ...\n" + context[-half:]


def _looks_truncated(original: str, fixed: str) -> bool:
    original_lines = original.count("\n") + 1
    fixed_lines = fixed.count("\n") + 1
    return original_lines >= 10 and fixed_lines < original_lines * 0.5
