"""Cloud model providers with free tiers (OpenAI-compatible APIs).

The client here has the same interface as OllamaClient, so the rest of TURBO
doesn't care where the model runs. Nothing runs on your computer except TURBO
itself, so it needs almost no RAM.

Costs: TURBO only picks free models, and if you never add a payment method to
the provider account there is nothing to charge. When the free allowance runs
out, requests are refused (HTTP 429) and TURBO says when to try again.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Iterator

import requests

from .llm import (
    CHARS_PER_TOKEN,
    LLMAuthError,
    LLMBadResponse,
    LLMConnectionError,
    LLMError,
    LLMInputTooLarge,
    LLMModelNotFound,
    LLMRateLimited,
    LLMTimeout,
)

PROVIDERS: dict[str, dict[str, Any]] = {
    "groq": {
        "label": "Groq",
        "base_url": "https://api.groq.com/openai/v1",
        "keys_url": "https://console.groq.com/keys",
        "env_key": "GROQ_API_KEY",
        "note": "Free, no card needed. Very fast. Roughly 1,000 requests a day per model.",
    },
    "gemini": {
        "label": "Google Gemini",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "keys_url": "https://aistudio.google.com/apikey",
        "env_key": "GEMINI_API_KEY",
        "note": "Free, no card needed. Google may use free-tier prompts to improve its products.",
    },
    "openrouter": {
        "label": "OpenRouter",
        "base_url": "https://openrouter.ai/api/v1",
        "keys_url": "https://openrouter.ai/keys",
        "env_key": "OPENROUTER_API_KEY",
        "note": "Free models only (':free'). Low daily limit (about 50 requests).",
        "free_only": True,
    },
}

# Model ids that aren't chat models.
NOT_CHAT = re.compile(r"whisper|tts|speech|audio|guard|embed|moderation|image|vision-only|"
                      r"playai|orpheus|transcri|rerank|compound|aqa|imagen|veo|learnlm", re.IGNORECASE)

# Preference order when picking models automatically (first match wins).
CODE_PREFERENCE = [r"qwen.*coder", r"devstral|codestral", r"qwen3", r"gpt-oss-120b", r"kimi",
                   r"deepseek", r"llama.*70b", r"gemini-[\d.]+-flash(?!-lite)", r"gpt-oss", r"gemini", r"qwen"]
CHAT_PREFERENCE = [r"gpt-oss-20b", r"gpt-oss", r"llama.*(8b|instant)", r"gemini-[\d.]+-flash-lite",
                   r"gemini-[\d.]+-flash", r"qwen3", r"llama", r"gemini", r"qwen", r"mistral"]


def provider_label(provider: str) -> str:
    return PROVIDERS.get(provider, {}).get("label", provider)


def usable_models(provider: str, model_ids: list[str]) -> list[str]:
    """Chat models that cost nothing on this provider."""
    free_only = PROVIDERS.get(provider, {}).get("free_only", False)
    usable = []
    for model_id in model_ids:
        if NOT_CHAT.search(model_id):
            continue
        if free_only and not model_id.endswith(":free"):
            continue
        usable.append(model_id)
    return sorted(usable)


def rank_models(provider: str, model_ids: list[str], role: str) -> list[str]:
    """All usable models, best first for this role ('chat' or 'code')."""
    usable = usable_models(provider, model_ids)
    ranked: list[str] = []
    for pattern in CODE_PREFERENCE if role == "code" else CHAT_PREFERENCE:
        for model_id in usable:
            if model_id not in ranked and re.search(pattern, model_id, re.IGNORECASE):
                ranked.append(model_id)
    return ranked + [m for m in usable if m not in ranked]


def pick_models(provider: str, model_ids: list[str]) -> tuple[str | None, str | None]:
    """(chat_model, code_model) chosen from what the provider offers."""
    usable = usable_models(provider, model_ids)

    def first(preferences: list[str]) -> str | None:
        for pattern in preferences:
            for model_id in usable:
                if re.search(pattern, model_id, re.IGNORECASE):
                    return model_id
        return usable[0] if usable else None

    return first(CHAT_PREFERENCE), first(CODE_PREFERENCE)


def _to_openai_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert TURBO's (Ollama-style) messages to the OpenAI format.

    OpenAI needs an id on each tool call and the same id on its result.
    """
    converted = []
    pending_ids: list[str] = []
    for index, message in enumerate(messages):
        role = message.get("role")
        if role == "assistant" and message.get("tool_calls"):
            calls = []
            for j, call in enumerate(message["tool_calls"]):
                function = call.get("function", {})
                args = function.get("arguments", {})
                call_id = call.get("id") or f"call_{index}_{j}"
                pending_ids.append(call_id)
                calls.append({"id": call_id, "type": "function", "function": {
                    "name": function.get("name", ""),
                    "arguments": args if isinstance(args, str) else json.dumps(args),
                }})
            converted.append({"role": "assistant", "content": message.get("content") or "",
                              "tool_calls": calls})
        elif role == "tool":
            call_id = pending_ids.pop(0) if pending_ids else f"call_orphan_{index}"
            converted.append({"role": "tool", "tool_call_id": call_id,
                              "content": message.get("content", "")})
        else:
            converted.append({"role": role, "content": message.get("content", "")})
    return converted


class OpenAICompatClient:
    """Talks to an OpenAI-compatible API (Groq, Gemini, OpenRouter, ...)."""

    def __init__(
        self,
        provider: str,
        api_key: str,
        base_url: str | None = None,
        timeout: float = 120,
        max_context_tokens: int = 32768,
        retries: int = 1,
        session: requests.Session | None = None,
    ):
        if not api_key:
            raise LLMAuthError(f"No API key for {provider_label(provider)}. Run: turbo setup")
        self.provider = provider
        self.label = provider_label(provider)
        self.base_url = (base_url or PROVIDERS.get(provider, {}).get("base_url", "")).rstrip("/")
        if not self.base_url:
            raise ValueError(f"Unknown provider {provider!r}. Set TURBO_API_BASE or run turbo setup.")
        self._api_key = api_key
        self.timeout = timeout
        self.max_context_tokens = max_context_tokens
        self.context_tokens = max_context_tokens  # reported by /status
        self.retries = retries
        self.http = session or requests.Session()
        self.last_done_reason: str | None = None
        self.is_cloud = True
        self.limits: dict[str, dict[str, Any]] = {}  # per model, from response headers

    def __repr__(self) -> str:  # never show the key
        return f"OpenAICompatClient(provider={self.provider!r}, base_url={self.base_url!r})"

    # ---- chat --------------------------------------------------------------------

    def _check_size(self, messages, expected_output_chars: int) -> None:
        chars = sum(len(str(m.get("content", ""))) for m in messages) + expected_output_chars
        needed = chars // CHARS_PER_TOKEN + 512
        if needed > self.max_context_tokens:
            raise LLMInputTooLarge(
                f"That's too much text (about {needed:,} tokens; TURBO's limit is "
                f"{self.max_context_tokens:,}). Select less code or run /clear."
            )

    def _payload(self, messages, model, temperature, stream, max_tokens, expected_output_chars=0,
                 tools=None) -> dict[str, Any]:
        self._check_size(messages, expected_output_chars)
        payload: dict[str, Any] = {
            "model": model,
            "messages": _to_openai_messages(messages),
            "temperature": temperature,
            "stream": stream,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens
        if tools:
            payload["tools"] = tools
        return payload

    def chat(self, messages, model, temperature=0.2, max_tokens=None, expected_output_chars=0) -> str:
        data = self._json(self._post("/chat/completions", self._payload(
            messages, model, temperature, False, max_tokens, expected_output_chars), model))
        choice = self._choice(data)
        self.last_done_reason = "length" if choice.get("finish_reason") == "length" else "stop"
        return (choice.get("message", {}).get("content") or "").strip()

    def chat_stream(self, messages, model, temperature=0.2, max_tokens=None,
                    expected_output_chars=0) -> Iterator[str]:
        self.last_done_reason = None
        response = self._post("/chat/completions", self._payload(
            messages, model, temperature, True, max_tokens, expected_output_chars), model, stream=True)
        try:
            for raw in response.iter_lines():
                if not raw:
                    continue
                line = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except ValueError as error:
                    raise LLMBadResponse(f"{self.label} sent a response TURBO couldn't read.") from error
                if chunk.get("error"):
                    raise LLMBadResponse(f"{self.label} error: {_message(chunk['error'])}")
                for choice in chunk.get("choices") or []:
                    piece = (choice.get("delta") or {}).get("content")
                    if piece:
                        yield piece
                    if choice.get("finish_reason"):
                        self.last_done_reason = "length" if choice["finish_reason"] == "length" else "stop"
        except requests.exceptions.Timeout as error:
            raise LLMTimeout(f"{self.label} stopped responding.") from error
        except requests.exceptions.RequestException as error:
            raise LLMConnectionError(f"Lost the connection to {self.label} mid-answer.") from error
        finally:
            response.close()

    def chat_tools(self, messages, model, tools, temperature=0.1, max_tokens=1024) -> dict[str, Any]:
        data = self._json(self._post("/chat/completions", self._payload(
            messages, model, temperature, False, max_tokens, tools=tools), model))
        message = self._choice(data).get("message") or {}
        calls = []
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            args = function.get("arguments") or "{}"
            try:
                parsed = json.loads(args) if isinstance(args, str) else args
            except ValueError:
                parsed = {}
            calls.append({"function": {"name": function.get("name", ""), "arguments": parsed}})
        return {"role": "assistant", "content": message.get("content") or "", "tool_calls": calls}

    # ---- models --------------------------------------------------------------------

    def model_details(self) -> list[dict[str, Any]]:
        data = self._json(self._get("/models"))
        ids = [str(m.get("id", "")).removeprefix("models/") for m in data.get("data", [])]
        return [{"name": name, "size": 0, "parameters": "", "quantization": ""}
                for name in usable_models(self.provider, ids)]

    def list_models(self) -> list[str]:
        return [m["name"] for m in self.model_details()]

    def loaded_models(self) -> list[dict[str, Any]]:
        return []  # nothing is loaded on this computer

    def warm(self, model: str) -> None:
        pass  # cloud models are always ready

    def supports_tools(self, name: str) -> bool | None:
        return True

    def pull(self, name, on_progress=None) -> None:
        raise LLMError(f"Downloading models is only for local Ollama. {self.label} models are "
                       "ready to use: see /models.")

    def unload(self, name: str) -> None:
        raise LLMError(f"Nothing to unload: {self.label} models run in the cloud, not on this computer.")

    # ---- HTTP ------------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}

    def _get(self, path: str) -> requests.Response:
        try:
            response = self.http.get(self.base_url + path, headers=self._headers(), timeout=15)
        except requests.exceptions.RequestException as error:
            raise LLMConnectionError(f"Can't reach {self.label}. Check your internet connection.") from error
        self._raise_for_status(response, model=None)
        return response

    def _post(self, path: str, payload: dict, model: str, stream: bool = False) -> requests.Response:
        attempt = 0
        while True:
            try:
                response = self.http.post(self.base_url + path, headers=self._headers(), json=payload,
                                          timeout=self.timeout, stream=stream)
                break
            except requests.exceptions.Timeout as error:
                raise LLMTimeout(f"{self.label} took longer than {self.timeout:.0f}s to answer.") from error
            except requests.exceptions.ConnectionError as error:
                if attempt < self.retries:
                    attempt += 1
                    time.sleep(0.5)
                    continue
                raise LLMConnectionError(
                    f"Can't reach {self.label}. Check your internet connection.") from error
        self._record_limits(model, response)
        self._raise_for_status(response, model)
        return response

    def _record_limits(self, model: str, response: requests.Response) -> None:
        """Remember the free-tier limits the provider reports in its headers."""
        headers = getattr(response, "headers", None) or {}
        try:
            if not any(str(k).lower().startswith("x-ratelimit") for k in headers):
                return
        except TypeError:
            return

        def number(name: str) -> int | None:
            value = headers.get(name)
            try:
                return int(float(value)) if value is not None else None
            except (TypeError, ValueError):
                return None

        self.limits[model] = {
            "limit_requests": number("x-ratelimit-limit-requests"),
            "remaining_requests": number("x-ratelimit-remaining-requests"),
            "reset_requests": parse_duration(headers.get("x-ratelimit-reset-requests")),
            "limit_tokens": number("x-ratelimit-limit-tokens"),
            "remaining_tokens": number("x-ratelimit-remaining-tokens"),
            "reset_tokens": parse_duration(headers.get("x-ratelimit-reset-tokens")),
            "at": time.monotonic(),
        }

    def _raise_for_status(self, response: requests.Response, model: str | None) -> None:
        status = response.status_code
        if status < 400:
            return
        detail = _error_detail(response)
        if status in (413, 400, 429):
            # "Request too large ... tokens per minute (TPM): Limit 8000, Requested 9200"
            sizes = re.search(r"Limit\s+(\d+),\s*Requested\s+(\d+)", detail)
            if sizes and (status == 413 or "per minute" in detail.lower() or "TPM" in detail):
                error = LLMInputTooLarge(
                    f"This request is about {int(sizes.group(2)):,} tokens, more than {model}'s free "
                    f"limit of {int(sizes.group(1)):,} tokens per minute. Selecting less code keeps "
                    "it within the free limit.")
                error.model, error.limit, error.requested = model, int(sizes.group(1)), int(sizes.group(2))
                raise error
        if status in (401, 403):
            raise LLMAuthError(f"{self.label} rejected the API key ({detail}). Run: turbo setup")
        if status == 402:
            raise LLMBadResponse(f"{model} on {self.label} needs paid credits. TURBO only uses free "
                                 "models: pick another with /models.")
        if status == 404:
            raise LLMModelNotFound(f"{self.label} doesn't offer '{model}'. See /models for what's available.")
        if status == 413 or (status == 400 and "context" in detail.lower()):
            raise LLMInputTooLarge(f"That's too much text for {model}. Select less code or run /clear.")
        if status == 429:
            wait = parse_duration(response.headers.get("retry-after"))
            if wait is None:
                info = self.limits.get(model, {})
                waits = [info.get("reset_requests") if info.get("remaining_requests") == 0 else None,
                         info.get("reset_tokens")]
                wait = max((w for w in waits if w is not None), default=None)
            when = f" Try again in about {_human_seconds(wait)}." if wait else " Try again later."
            error = LLMRateLimited(f"{model} reached {self.label}'s free limit ({detail}).{when} "
                                   "Or switch to another model with /model; limits are per model.")
            error.model, error.retry_after = model, wait
            raise error
        raise LLMBadResponse(f"{self.label} returned HTTP {status}: {detail}")

    def _json(self, response: requests.Response) -> dict[str, Any]:
        try:
            data = response.json()
        except ValueError as error:
            raise LLMBadResponse(f"{self.label} sent a response TURBO couldn't read.") from error
        if not isinstance(data, dict):
            raise LLMBadResponse(f"{self.label} sent a response TURBO couldn't read.")
        return data

    def _choice(self, data: dict[str, Any]) -> dict[str, Any]:
        choices = data.get("choices") or []
        if not choices or not isinstance(choices[0], dict):
            raise LLMBadResponse(f"{self.label} sent an empty answer.")
        return choices[0]


def _message(error: Any) -> str:
    if isinstance(error, dict):
        return str(error.get("message") or error)[:300]
    return str(error)[:300]


def _error_detail(response: requests.Response) -> str:
    try:
        data = response.json()
    except ValueError:
        return response.text[:200] or "no details"
    if isinstance(data, list) and data:
        data = data[0]
    if isinstance(data, dict) and "error" in data:
        return _message(data["error"])
    return str(data)[:200]


def parse_duration(value: Any) -> float | None:
    """Seconds from '120', '1.5', '547ms', '4m19.2s' or '1h2m3s' (None if unknown)."""
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        pass
    parts = re.findall(r"(\d+(?:\.\d+)?)(ms|h|m|s)", text)
    if not parts or "".join(n + u for n, u in parts) != text.replace(" ", ""):
        return None
    scale = {"h": 3600, "m": 60, "s": 1, "ms": 0.001}
    return sum(float(n) * scale[u] for n, u in parts)


def _human_seconds(value: Any) -> str:
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return str(value)
    if seconds < 90:
        return f"{seconds:.0f} seconds"
    if seconds < 5400:
        return f"{seconds / 60:.0f} minutes"
    return f"{seconds / 3600:.1f} hours"
