"""Fallback chain for free cloud models.

When a model hits its free limit, TURBO doesn't just fail. It works out what
the user can do instead, with estimated times, and asks:

    qwen/qwen3.8-27b reached its free limit (resets in about 35 seconds).
      1. Continue now with openai/gpt-oss-120b      ~3s
      2. Wait for qwen/qwen3.8-27b                  ~38s
      3. Cancel

The choice is remembered until the first model's limit resets; after that
TURBO goes back to it automatically. Requests that are too big for the free
per-minute token limit are caught before sending, with a recommendation to
select less code.

The chain is: the configured model, then the provider's other free models
(best first for the task), then local Ollama if it's installed and running.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

from .cloud import CHARS_PER_TOKEN, rank_models
from .llm import LLMError, LLMInputTooLarge, LLMRateLimited

DEFAULT_SECONDS = {"cloud": 3.0, "local": 25.0}  # until we've measured
MAX_WAIT_OPTION = 15 * 60  # don't offer to wait longer than this
LOCAL_DEFAULTS = {"chat": "llama3.2:3b", "code": "qwen2.5-coder:3b"}


class RequestCancelled(LLMError):
    pass


@dataclass
class Option:
    kind: str                 # "switch", "wait" or "cancel"
    label: str
    estimate_seconds: float = 0.0
    wait_seconds: float = 0.0
    model: str | None = None
    backend: str = "cloud"    # "cloud" or "local"

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "label": self.label, "model": self.model, "backend": self.backend,
                "estimate_seconds": round(self.estimate_seconds, 1),
                "wait_seconds": round(self.wait_seconds, 1)}


@dataclass
class Decision:
    message: str
    options: list[Option]
    requested_model: str
    kind: str = "limit"       # "limit" or "too_large"
    advice: str = ""
    failed_model: str = ""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "kind": self.kind, "message": self.message, "advice": self.advice,
                "options": [o.to_dict() for o in self.options]}


class LimitReached(LLMRateLimited):
    """The request can't run as-is; decision says what the user can do."""

    def __init__(self, decision: Decision):
        text = decision.message + (f" {decision.advice}" if decision.advice else "")
        super().__init__(text)
        self.decision = decision


def format_seconds(seconds: float) -> str:
    seconds = max(0.0, seconds)
    if seconds < 1:
        return "<1s"
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


def estimate_tokens(messages: list[dict[str, Any]], expected_output_chars: int = 0,
                    max_tokens: int | None = None) -> int:
    chars = sum(len(str(m.get("content", ""))) for m in messages)
    # The answer usually uses far less than max_tokens allows, so count a
    # realistic size (expected output, or up to 1024 tokens) instead.
    output = max(expected_output_chars // CHARS_PER_TOKEN, min(max_tokens or 0, 1024), 300)
    return chars // CHARS_PER_TOKEN + output


class FallbackClient:
    """Wraps a cloud client (OpenAICompatClient) with the fallback chain."""

    def __init__(self, primary, local_factory: Callable[[], Any] | None = None,
                 clock: Callable[[], float] = time.monotonic):
        self.primary = primary
        self.local_factory = local_factory
        self.clock = clock
        self.settings_getter: Callable[[], Any] | None = None
        self.cooldowns: dict[str, float] = {}           # model -> time its limit resets
        self.routes: dict[str, tuple[str, str, float]] = {}  # requested -> (backend, model, until)
        self.one_shot: dict[str, tuple[str, str]] = {}  # requested -> (backend, model) for one request
        self.speeds: dict[tuple[str, str], float] = {}  # (backend, model) -> seconds per answer
        self.decisions: dict[str, Decision] = {}        # recent decisions, for the VS Code server
        self.last_route_note: str | None = None
        self._last_used = primary
        self._local: tuple[Any, list[str], float] | None = None
        self._models_cache: tuple[list[str], float] | None = None

    def bind(self, settings_getter: Callable[[], Any]) -> None:
        self.settings_getter = settings_getter

    def __getattr__(self, name: str):
        # Everything else (list_models, model_details, label, provider, ...)
        # comes from the primary cloud client.
        return getattr(self.primary, name)

    @property
    def last_done_reason(self) -> str | None:
        return getattr(self._last_used, "last_done_reason", None)

    # ---- the three ways of talking to a model ---------------------------------------

    def chat(self, messages, model, temperature=0.2, max_tokens=None, expected_output_chars=0) -> str:
        client, target, backend = self._prepare(messages, model, expected_output_chars, max_tokens)
        return self._timed(backend, target, lambda: client.chat(
            messages, model=target, temperature=temperature, max_tokens=max_tokens,
            expected_output_chars=expected_output_chars), model)

    def chat_tools(self, messages, model, tools, temperature=0.1, max_tokens=1024) -> dict[str, Any]:
        client, target, backend = self._prepare(messages, model, 0, max_tokens)
        return self._timed(backend, target, lambda: client.chat_tools(
            messages, target, tools, temperature=temperature, max_tokens=max_tokens), model)

    def chat_stream(self, messages, model, temperature=0.2, max_tokens=None,
                    expected_output_chars=0) -> Iterator[str]:
        client, target, backend = self._prepare(messages, model, expected_output_chars, max_tokens)
        start = self.clock()
        pieces = client.chat_stream(messages, model=target, temperature=temperature,
                                    max_tokens=max_tokens, expected_output_chars=expected_output_chars)
        try:
            # Limit errors arrive before the first word, so nothing has been
            # shown yet and the request can be redone on another model.
            first = next(pieces)
        except StopIteration:
            self._record_speed(backend, target, self.clock() - start)
            return
        except (LLMRateLimited, LLMInputTooLarge) as error:
            raise self._limit_decision(model, target, backend, error, messages, expected_output_chars,
                                       max_tokens) from error
        yield first
        yield from pieces
        self._record_speed(backend, target, self.clock() - start)

    # ---- choosing --------------------------------------------------------------------

    def choose(self, decision: Decision | str, option: Option | int | str) -> Option:
        """Apply the user's choice. Accepts a Decision or its id, and an Option,
        its index or its kind."""
        if isinstance(decision, str):
            if decision not in self.decisions:
                raise LLMError("That choice has expired. Send the request again.")
            decision = self.decisions[decision]
        if isinstance(option, int):
            option = decision.options[option]
        elif isinstance(option, str):
            option = next((o for o in decision.options if o.kind == option), None) or \
                decision.options[0]
        if option.kind == "cancel":
            raise RequestCancelled("Cancelled.")
        if option.kind == "switch":
            if decision.kind == "too_large":
                self.one_shot[decision.requested_model] = (option.backend, option.model)
            else:
                # Use the alternative until the model the user asked for is free again.
                resets = [self.cooldowns.get(m, 0) for m in (decision.requested_model, decision.failed_model)]
                until = max(resets) if max(resets) > self.clock() else self.clock() + 60
                self.routes[decision.requested_model] = (option.backend, option.model, until)
        return option

    # ---- status -------------------------------------------------------------------------

    def limits_report(self) -> list[dict[str, Any]]:
        now = self.clock()
        report = []
        for model, info in getattr(self.primary, "limits", {}).items():
            cooling = self.cooldowns.get(model, 0) - now
            report.append({**{k: v for k, v in info.items() if k != "at"}, "model": model,
                           "cooling_seconds": max(0.0, cooling)})
        return report

    # ---- internals ------------------------------------------------------------------------

    def _role(self, model: str) -> str:
        settings = self.settings_getter() if self.settings_getter else None
        return "code" if settings is not None and model == settings.code_model else "chat"

    def _client_for(self, backend: str):
        if backend == "local":
            local = self._local_status()
            if local is None:
                raise LLMError("Local Ollama isn't available any more. Start the Ollama app.")
            return local[0]
        return self.primary

    def _prepare(self, messages, model, expected_output_chars, max_tokens):
        """Pick where this request goes; raise LimitReached if the user must choose."""
        self.last_route_note = None
        now = self.clock()
        if model in self.one_shot:
            backend, target = self.one_shot.pop(model)
            self.last_route_note = f"answered by {target}" + (" on this computer" if backend == "local" else "")
            return self._client_for(backend), target, backend

        route = self.routes.get(model)
        if route and now < route[2]:
            backend, target, until = route
            if backend == "cloud" and self.cooldowns.get(target, 0) > now:
                del self.routes[model]  # the fallback is at its limit too
            else:
                self.last_route_note = (f"answered by {target} while {model} resets "
                                        f"(about {format_seconds(until - now)} left)")
                return self._client_for(backend), target, backend
        elif route:
            del self.routes[model]
            self.last_route_note = f"back to {model}"

        # Known to be at its limit: ask now instead of spending a request.
        if self.cooldowns.get(model, 0) > now:
            raise self._limit_decision(model, model, "cloud", None, messages, expected_output_chars,
                                       max_tokens)

        # Too big for the free per-minute token limit: say so before sending.
        limit = self._token_limit(model)
        needed = estimate_tokens(messages, expected_output_chars, max_tokens)
        if limit and needed > limit:
            raise LimitReached(self._too_large_decision(model, needed, limit))
        return self.primary, model, "cloud"

    def _token_limit(self, model: str) -> int | None:
        """Per-minute token limit for a model. Until the provider has told us
        (in its response headers), assume the lowest limit seen on this
        account: free-tier limits are similar across models."""
        limits = getattr(self.primary, "limits", {})
        known = (limits.get(model) or {}).get("limit_tokens")
        if known:
            return known
        seen = [info["limit_tokens"] for info in limits.values() if info.get("limit_tokens")]
        return min(seen) if seen else None

    def _timed(self, backend: str, target: str, call: Callable[[], Any], requested: str):
        start = self.clock()
        client = self._client_for(backend)
        try:
            result = call()
        except (LLMRateLimited, LLMInputTooLarge) as error:
            if isinstance(error, LimitReached):
                raise
            raise self._limit_decision(requested, target, backend, error, None, 0, None) from error
        self._last_used = client
        self._record_speed(backend, target, self.clock() - start)
        return result

    def _record_speed(self, backend: str, model: str, seconds: float) -> None:
        key = (backend, model)
        previous = self.speeds.get(key)
        self.speeds[key] = seconds if previous is None else previous * 0.6 + seconds * 0.4
        if backend == "local" and self._local:
            self._last_used = self._local[0]
        elif backend == "cloud":
            self._last_used = self.primary

    def _estimate(self, backend: str, model: str) -> float:
        return self.speeds.get((backend, model), DEFAULT_SECONDS[backend])

    def _cloud_models(self) -> list[str]:
        now = self.clock()
        if self._models_cache and now - self._models_cache[1] < 300:
            return self._models_cache[0]
        try:
            models = self.primary.list_models()
        except LLMError:
            models = []
        self._models_cache = (models, now)
        return models

    def _local_status(self) -> tuple[Any, list[str], float] | None:
        """(client, installed models, checked_at) if local Ollama is usable."""
        now = self.clock()
        if self._local and now - self._local[2] < 60:
            return self._local
        if self.local_factory is None:
            return None
        try:
            client = self.local_factory()
            models = client.list_models()
        except Exception:  # noqa: BLE001 - any failure just means "not available"
            self._local = None
            return None
        self._local = (client, models, now) if models else None
        return self._local

    def _local_model(self, role: str) -> str | None:
        local = self._local_status()
        if local is None:
            return None
        installed = local[1]
        preferred = LOCAL_DEFAULTS[role]
        if preferred in installed:
            return preferred
        other = LOCAL_DEFAULTS["code" if role == "chat" else "chat"]
        return other if other in installed else installed[0]

    def _alternatives(self, requested: str, exclude: set[str], needed_tokens: int | None) -> list[Option]:
        """Other places this request could go right now, best first."""
        now = self.clock()
        role = self._role(requested)
        provider = getattr(self.primary, "provider", "groq")
        options = []
        for model in rank_models(provider, self._cloud_models(), role):
            if model in exclude or self.cooldowns.get(model, 0) > now:
                continue
            limit = self._token_limit(model)
            if needed_tokens and limit and needed_tokens > limit:
                continue
            options.append(Option("switch", f"Continue now with {model}",
                                  self._estimate("cloud", model), model=model))
            break  # the best available cloud model is enough to offer
        local_model = self._local_model(role)
        if local_model:
            options.append(Option("switch", f"Continue now with {local_model} on this computer "
                                            "(free, no limit, slower, uses 2-3 GB of RAM)",
                                  self._estimate("local", local_model), model=local_model, backend="local"))
        return options

    def _limit_decision(self, requested, target, backend, error, messages, expected_output_chars,
                        max_tokens) -> LimitReached:
        if isinstance(error, LLMInputTooLarge):
            needed = getattr(error, "requested", None) or (
                estimate_tokens(messages, expected_output_chars, max_tokens) if messages else 0)
            limit = getattr(error, "limit", None) or needed
            if getattr(error, "limit", None) and backend == "cloud":
                # Remember what the provider told us, for the next pre-check.
                getattr(self.primary, "limits", {}).setdefault(target, {})["limit_tokens"] = error.limit
            return LimitReached(self._too_large_decision(requested, needed, limit, failed=target))

        now = self.clock()
        wait = getattr(error, "retry_after", None) if error is not None else None
        if error is not None:
            self.cooldowns[target] = now + (wait if wait is not None else 60)
        wait = max(0.0, self.cooldowns.get(target, now) - now)

        options = self._alternatives(requested, exclude={target, requested}, needed_tokens=None)
        # Offer to wait for whichever of the two frees up first.
        candidates = [(max(0.0, self.cooldowns.get(m, now) - now), m) for m in {requested, target}]
        soonest, wait_model = min(candidates)
        wait_backend = "cloud" if wait_model == requested else backend
        if soonest <= MAX_WAIT_OPTION:
            options.append(Option("wait", f"Wait for {wait_model}",
                                  soonest + self._estimate(wait_backend, wait_model),
                                  wait_seconds=soonest, model=wait_model, backend=wait_backend))
        options.append(Option("cancel", "Cancel"))
        decision = Decision(
            message=f"{target} reached its free limit (resets in about {format_seconds(wait)}).",
            options=options, requested_model=requested, failed_model=target)
        self._remember(decision)
        return LimitReached(decision)

    def _too_large_decision(self, requested: str, needed: int, limit: int, failed: str | None = None) -> Decision:
        failed = failed or requested
        options = self._alternatives(requested, exclude={failed}, needed_tokens=needed)
        options.append(Option("cancel", "Cancel (then select less code and try again)"))
        decision = Decision(
            kind="too_large",
            message=(f"This request is about {needed:,} tokens, more than {failed}'s free limit of "
                     f"{limit:,} tokens per minute."),
            advice="Selecting less code (just the part you need help with) keeps requests within "
                   "the free limit and gets faster answers.",
            options=options, requested_model=requested, failed_model=failed)
        self._remember(decision)
        return decision

    def _remember(self, decision: Decision) -> None:
        self.decisions[decision.id] = decision
        while len(self.decisions) > 20:
            self.decisions.pop(next(iter(self.decisions)))


def run_with_choices(call: Callable[[], Any], llm, ask: Callable[[Decision], Option | None],
                     wait: Callable[[float, str], None]):
    """Run call(); when a limit is reached, ask the user and retry as chosen."""
    while True:
        try:
            return call()
        except LimitReached as stop:
            option = ask(stop.decision)
            if option is None:
                raise RequestCancelled("Cancelled.") from None
            option = llm.choose(stop.decision, option)
            if option.kind == "wait" and option.wait_seconds > 0:
                wait(option.wait_seconds, option.model or "")
