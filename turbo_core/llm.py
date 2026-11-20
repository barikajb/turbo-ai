"""Minimal Ollama chat client with clear, user-facing errors."""

from __future__ import annotations

import time
from typing import Any

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

    def chat(self, messages: list[dict[str, str]], model: str, temperature: float = 0.2) -> str:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            # Ollama's default context window is small and silently truncates
            # long prompts, which would cut off pasted files.
            "options": {"num_ctx": self.context_tokens, "temperature": temperature},
        }

        attempt = 0
        while True:
            try:
                response = self.http.post(
                    f"{self.base_url}/api/chat", json=payload, timeout=self.timeout
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

        try:
            content = response.json()["message"]["content"]
        except (ValueError, KeyError, TypeError) as error:
            raise LLMBadResponse("Ollama sent a response TURBO couldn't read.") from error

        return content.strip()

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
