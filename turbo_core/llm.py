"""Minimal Ollama chat client with clear, user-facing errors."""

from __future__ import annotations

import json
import time
from typing import Any, Iterator

import requests


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


class OllamaClient:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout: float = 300,
        context_tokens: int = 8192,
        retries: int = 1,
        session: requests.Session | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.context_tokens = context_tokens
        self.retries = retries
        self.http = session or requests.Session()

    def _payload(self, messages, model, temperature, stream: bool) -> dict[str, Any]:
        return {
            "model": model,
            "messages": messages,
            "stream": stream,
            # Ollama's default context window is small and silently truncates
            # long prompts, which would cut off pasted files.
            "options": {"num_ctx": self.context_tokens, "temperature": temperature},
        }

    def chat(self, messages: list[dict[str, str]], model: str, temperature: float = 0.2) -> str:
        response = self._post(self._payload(messages, model, temperature, stream=False), model)
        try:
            content = response.json()["message"]["content"]
        except (ValueError, KeyError, TypeError) as error:
            raise LLMBadResponse("Ollama sent a response TURBO couldn't read.") from error
        return content.strip()

    def chat_stream(
        self, messages: list[dict[str, str]], model: str, temperature: float = 0.2
    ) -> Iterator[str]:
        """Yield the reply in pieces as the model writes it.

        The timeout applies to the gap between pieces, not the whole answer.
        """
        response = self._post(
            self._payload(messages, model, temperature, stream=True), model, stream=True
        )
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
                    return
        except requests.exceptions.Timeout as error:
            raise LLMTimeout(
                f"The model stopped responding for {self.timeout:.0f}s. Raise TURBO_TIMEOUT if needed."
            ) from error
        except requests.exceptions.RequestException as error:
            raise LLMConnectionError("Lost the connection to Ollama mid-answer.") from error
        finally:
            response.close()

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
                    "Try a smaller selection or raise TURBO_TIMEOUT."
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

    def list_models(self) -> list[str]:
        try:
            response = self.http.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            return [m["name"] for m in response.json().get("models", [])]
        except (requests.exceptions.RequestException, ValueError, KeyError) as error:
            raise LLMConnectionError(f"Can't reach Ollama at {self.base_url}.") from error


def _error_detail(response: requests.Response) -> str:
    try:
        return str(response.json().get("error", response.text))[:300]
    except ValueError:
        return response.text[:300]
