"""Ollama client with clear, user-facing errors and speed-minded defaults.

Speed notes (CPU machines especially):
- num_ctx: a big context window costs RAM and time even when unused, so we
  start small and only grow it when a long file needs it. Changing num_ctx
  makes Ollama reload the model, so it only ever grows ("sticky").
- keep_alive: keeps the model loaded between questions (reloading costs ~10s).
- num_predict: caps answer length for chat so replies can't run on forever.
"""

from __future__ import annotations

import json
import math
import time
from typing import Any, Callable, Iterator

import requests

CHARS_PER_TOKEN = 3  # conservative estimate for code-heavy text


class LLMError(Exception):
    """Base class for model errors. The message is safe to show the user."""


class LLMConnectionError(LLMError):
    pass


class LLMModelNotFound(LLMError):
    pass


class LLMTimeout(LLMError):
    pass


class LLMBadResponse(LLMError):
    pass


class LLMInputTooLarge(LLMError):
    pass


class LLMAuthError(LLMError):
    pass


class LLMRateLimited(LLMError):
    pass


class OllamaClient:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout: float = 300,
        context_tokens: int = 4096,
        max_context_tokens: int = 16384,
        keep_alive: str = "30m",
        retries: int = 1,
        session: requests.Session | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.context_tokens = context_tokens
        self.max_context_tokens = max(max_context_tokens, context_tokens)
        self.keep_alive = keep_alive
        self.retries = retries
        self.http = session or requests.Session()
        self.last_done_reason: str | None = None

    # ---- context sizing ----------------------------------------------------

    def context_for(self, messages: list[dict[str, Any]], expected_output_chars: int = 0) -> int:
        """Context window big enough for these messages plus the answer.

        Grows in 2048-token steps and never shrinks, so Ollama doesn't reload
        the model back and forth.
        """
        prompt_chars = sum(len(str(m.get("content", ""))) for m in messages)
        needed = (prompt_chars + expected_output_chars) // CHARS_PER_TOKEN + 512
        if needed > self.max_context_tokens:
            raise LLMInputTooLarge(
                f"That's too much text for the model (about {needed:,} tokens; the limit is "
                f"{self.max_context_tokens:,}). Select less code, run /clear to drop old "
                "conversation, or raise TURBO_MAX_CONTEXT_TOKENS."
            )
        if needed > self.context_tokens:
            self.context_tokens = min(self.max_context_tokens, math.ceil(needed / 2048) * 2048)
        return self.context_tokens

    def _payload(
        self,
        messages,
        model,
        temperature,
        stream: bool,
        max_tokens: int | None,
        expected_output_chars: int = 0,
        tools: list[dict] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "keep_alive": self.keep_alive,
            "options": {
                "num_ctx": self.context_for(messages, expected_output_chars),
                "temperature": temperature,
                "num_predict": max_tokens if max_tokens else -1,
            },
        }
        if tools:
            payload["tools"] = tools
        return payload

    # ---- chat ----------------------------------------------------------------

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        expected_output_chars: int = 0,
    ) -> str:
        payload = self._payload(messages, model, temperature, False, max_tokens, expected_output_chars)
        data = self._json(self._post(payload, model))
        self.last_done_reason = data.get("done_reason")
        try:
            return data["message"]["content"].strip()
        except (KeyError, TypeError, AttributeError) as error:
            raise LLMBadResponse("Ollama sent a response TURBO couldn't read.") from error

    def chat_stream(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        expected_output_chars: int = 0,
    ) -> Iterator[str]:
        """Yield the reply in pieces as the model writes it.

        The timeout applies to the gap between pieces, not the whole answer.
        """
        self.last_done_reason = None
        payload = self._payload(messages, model, temperature, True, max_tokens, expected_output_chars)
        response = self._post(payload, model, stream=True)
        try:
            for line in response.iter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except ValueError as error:
                    raise LLMBadResponse("Ollama sent a response TURBO couldn't read.") from error
                if chunk.get("error"):
                    raise LLMBadResponse(f"Ollama error: {chunk['error']}")
                piece = (chunk.get("message") or {}).get("content", "")
                if piece:
                    yield piece
                if chunk.get("done"):
                    self.last_done_reason = chunk.get("done_reason")
                    return
        except requests.exceptions.Timeout as error:
            raise LLMTimeout(
                f"The model stopped responding for {self.timeout:.0f}s. Raise TURBO_TIMEOUT if needed."
            ) from error
        except requests.exceptions.RequestException as error:
            raise LLMConnectionError("Lost the connection to Ollama mid-answer.") from error
        finally:
            response.close()

    def chat_tools(
        self,
        messages: list[dict[str, Any]],
        model: str,
        tools: list[dict],
        temperature: float = 0.1,
        max_tokens: int | None = 1024,
    ) -> dict[str, Any]:
        """One non-streamed turn with tool definitions. Returns the message
        dict, which may contain 'tool_calls'."""
        payload = self._payload(messages, model, temperature, False, max_tokens, tools=tools)
        data = self._json(self._post(payload, model))
        message = data.get("message")
        if not isinstance(message, dict):
            raise LLMBadResponse("Ollama sent a response TURBO couldn't read.")
        return message

    def warm(self, model: str) -> None:
        """Load a model into memory ahead of the first question. Never raises."""
        try:
            self.http.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": model,
                    "messages": [],
                    "keep_alive": self.keep_alive,
                    "options": {"num_ctx": self.context_tokens},
                },
                timeout=self.timeout,
            )
        except requests.exceptions.RequestException:
            pass

    # ---- model management ------------------------------------------------------

    def list_models(self) -> list[str]:
        return [m["name"] for m in self.model_details()]

    def model_details(self) -> list[dict[str, Any]]:
        """Installed models: name, size (bytes), parameter size, quantization."""
        data = self._get("/api/tags")
        return [
            {
                "name": m["name"],
                "size": m.get("size", 0),
                "parameters": (m.get("details") or {}).get("parameter_size", ""),
                "quantization": (m.get("details") or {}).get("quantization_level", ""),
            }
            for m in data.get("models", [])
        ]

    def loaded_models(self) -> list[dict[str, Any]]:
        """Models currently in memory, with how much of each is on the GPU."""
        data = self._get("/api/ps")
        loaded = []
        for m in data.get("models", []):
            size = m.get("size") or 0
            vram = m.get("size_vram") or 0
            gpu_share = (vram / size) if size else 0.0
            loaded.append({
                "name": m.get("name", "?"),
                "size": size,
                "gpu_percent": round(gpu_share * 100),
                "context": m.get("context_length"),
                "expires_at": m.get("expires_at"),
            })
        return loaded

    def unload(self, name: str) -> None:
        """Remove a model from memory now (frees RAM; it stays installed)."""
        try:
            response = self.http.post(
                f"{self.base_url}/api/generate", json={"model": name, "keep_alive": 0}, timeout=30
            )
        except requests.exceptions.RequestException as error:
            raise LLMConnectionError(f"Can't reach Ollama at {self.base_url}.") from error
        if response.status_code == 404:
            raise LLMModelNotFound(f"Model '{name}' is not installed.")
        if response.status_code >= 400:
            raise LLMBadResponse(f"Couldn't unload {name}: {_error_detail(response)}")

    def supports_tools(self, name: str) -> bool | None:
        """Whether Ollama says this model can call tools (None if unknown)."""
        try:
            response = self.http.post(f"{self.base_url}/api/show", json={"model": name}, timeout=10)
            response.raise_for_status()
            capabilities = response.json().get("capabilities")
        except (requests.exceptions.RequestException, ValueError):
            return None
        return None if capabilities is None else "tools" in capabilities

    def pull(self, name: str, on_progress: Callable[[str, int, int], None] | None = None) -> None:
        """Download a model. on_progress(status, completed_bytes, total_bytes)."""
        try:
            response = self.http.post(
                f"{self.base_url}/api/pull", json={"model": name, "stream": True},
                timeout=self.timeout, stream=True,
            )
        except requests.exceptions.RequestException as error:
            raise LLMConnectionError(f"Can't reach Ollama at {self.base_url}.") from error
        if response.status_code >= 400:
            raise LLMBadResponse(f"Couldn't download {name}: {_error_detail(response)}")
        try:
            for line in response.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                if chunk.get("error"):
                    raise LLMBadResponse(f"Couldn't download {name}: {chunk['error']}")
                if on_progress:
                    on_progress(chunk.get("status", ""), chunk.get("completed", 0), chunk.get("total", 0))
        except requests.exceptions.RequestException as error:
            raise LLMConnectionError(f"Lost the connection to Ollama while downloading {name}.") from error
        except ValueError as error:
            raise LLMBadResponse("Ollama sent a response TURBO couldn't read.") from error
        finally:
            response.close()

    # ---- HTTP helpers ------------------------------------------------------------

    def _get(self, path: str) -> dict[str, Any]:
        try:
            response = self.http.get(f"{self.base_url}{path}", timeout=5)
            response.raise_for_status()
            return response.json()
        except (requests.exceptions.RequestException, ValueError) as error:
            raise LLMConnectionError(f"Can't reach Ollama at {self.base_url}.") from error

    @staticmethod
    def _json(response: requests.Response) -> dict[str, Any]:
        try:
            data = response.json()
        except ValueError as error:
            raise LLMBadResponse("Ollama sent a response TURBO couldn't read.") from error
        if not isinstance(data, dict):
            raise LLMBadResponse("Ollama sent a response TURBO couldn't read.")
        return data

    def _post(self, payload: dict[str, Any], model: str, stream: bool = False) -> requests.Response:
        attempt = 0
        while True:
            try:
                response = self.http.post(
                    f"{self.base_url}/api/chat", json=payload, timeout=self.timeout, stream=stream
                )
                break
            except requests.exceptions.Timeout as error:
                raise LLMTimeout(
                    f"The model took longer than {self.timeout:.0f}s to answer. "
                    "Try a smaller selection, a smaller model (/model), or raise TURBO_TIMEOUT."
                ) from error
            except requests.exceptions.ConnectionError as error:
                if attempt < self.retries:
                    attempt += 1
                    time.sleep(0.5)
                    continue
                raise LLMConnectionError(
                    f"Can't reach Ollama at {self.base_url}. Is Ollama running? "
                    "(Start it, or run `ollama serve`.)"
                ) from error

        if response.status_code == 404:
            raise LLMModelNotFound(
                f"Model '{model}' is not installed. Run: ollama pull {model}"
            )
        if response.status_code >= 400:
            detail = _error_detail(response)
            raise LLMBadResponse(f"Ollama returned HTTP {response.status_code}: {detail}")
        return response


def _error_detail(response: requests.Response) -> str:
    try:
        return str(response.json().get("error", response.text))[:300]
    except ValueError:
        return response.text[:300]
