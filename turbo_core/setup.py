"""`turbo setup`: choose where the models run.

Cloud (free tier, almost no RAM on this computer) or local Ollama. The API
key is typed hidden and stored only in ~/.turbo/.env on this computer.
"""

from __future__ import annotations

import getpass
import os
import sys
import time

from .cloud import PROVIDERS, OpenAICompatClient, pick_models, provider_label, usable_models
from .config import Settings, save_setting, turbo_home
from .llm import LLMAuthError, LLMError

LOCAL_DEFAULTS = {"chat": "llama3.2:3b", "code": "qwen2.5-coder:3b"}
CHOICES = ["groq", "gemini", "openrouter", "ollama"]


def run_setup(args, input_func=input, secret_func=getpass.getpass, out=print) -> int:
    if not sys.stdin.isatty() and input_func is input:
        out("turbo setup asks questions, so run it in an interactive terminal.")
        return 2

    current = Settings.from_env()
    out("TURBO setup: where should the AI models run?\n")
    provider = args.provider if getattr(args, "provider", None) else _choose(current, input_func, out)
    if provider is None:
        out("Nothing changed.")
        return 0
    if provider == "ollama":
        return _setup_ollama(out)
    return _setup_cloud(provider, current, input_func, secret_func, out)


def _choose(current: Settings, input_func, out) -> str | None:
    rows = [
        ("groq", "Groq (recommended)", PROVIDERS["groq"]["note"]),
        ("gemini", "Google Gemini", PROVIDERS["gemini"]["note"]),
        ("openrouter", "OpenRouter", PROVIDERS["openrouter"]["note"]),
        ("ollama", "Local Ollama", "Runs on this computer: private and offline, but needs 2-5 GB of RAM."),
    ]
    for number, (key, title, note) in enumerate(rows, 1):
        mark = "  (current)" if key == current.provider else ""
        out(f"  {number}. {title}{mark}\n     {note}")
    out("")
    answer = input_func("Choose 1-4 [1]: ").strip() or "1"
    if answer.lower() in ("q", "quit", "n", "no"):
        return None
    if answer.isdigit() and 1 <= int(answer) <= len(rows):
        return rows[int(answer) - 1][0]
    if answer.lower() in CHOICES:
        return answer.lower()
    out(f"Didn't understand {answer!r}.")
    return None


def _setup_cloud(provider: str, current: Settings, input_func, secret_func, out) -> int:
    info = PROVIDERS[provider]
    label = info["label"]
    out(f"\n{label}: get a free API key at {info['keys_url']}")
    out("  - Sign up for free. Do NOT add a credit card or payment method:")
    out("    without one there is nothing to charge, and TURBO only uses free models.")
    out(f"  - The key is saved only on this computer, in {turbo_home() / '.env'}.\n")

    existing = current.api_key if current.provider == provider else ""
    for attempt in range(3):
        prompt = "Paste your API key (hidden; press Enter to keep the current one): " if existing \
            else "Paste your API key (hidden): "
        key = secret_func(prompt).strip() or existing
        if not key:
            out("No key entered. Nothing changed.")
            return 1
        try:
            client = OpenAICompatClient(provider, key, timeout=60)
            models = client.list_models()
        except LLMAuthError:
            out(f"{label} didn't accept that key. Copy it again from {info['keys_url']}.")
            existing = ""
            continue
        except LLMError as error:
            out(f"Couldn't check the key: {error}")
            return 1
        break
    else:
        out("Giving up after 3 tries. Nothing changed.")
        return 1

    if not models:
        out(f"{label} didn't list any free chat models for this key. Nothing changed.")
        return 1
    chat, code = pick_models(provider, models)
    out(f"\n✓ Key works. {len(models)} free models available.")
    out(f"  chat (questions):     {chat}")
    out(f"  code (explain / fix): {code}")

    out("\nTesting a short request...")
    started = time.monotonic()
    try:
        reply = client.chat([{"role": "user", "content": "Reply with exactly: ok"}],
                            model=chat, max_tokens=10)
    except LLMError as error:
        out(f"The test request failed: {error}\nNothing changed.")
        return 1
    out(f"✓ {label} answered {reply[:20]!r} in {time.monotonic() - started:.1f}s.")

    path = save_setting("TURBO_PROVIDER", provider)
    save_setting("TURBO_API_KEY", key)
    save_setting("TURBO_CHAT_MODEL", chat)
    save_setting("TURBO_CODE_MODEL", code)
    _restrict_permissions(path)
    out(f"\nSaved to {path}. Keep that file private: it contains your API key.")
    out("Switch models any time with /model (list them with /models).")
    out("\nAll set. Run: turbo")
    return 0


def _setup_ollama(out) -> int:
    path = save_setting("TURBO_PROVIDER", "ollama")
    save_setting("TURBO_API_KEY", "")
    save_setting("TURBO_CHAT_MODEL", LOCAL_DEFAULTS["chat"])
    save_setting("TURBO_CODE_MODEL", LOCAL_DEFAULTS["code"])
    out(f"\nSaved to {path}. TURBO will use Ollama on this computer.")
    out("If you haven't yet: install Ollama from https://ollama.com, then run")
    out(f"  ollama pull {LOCAL_DEFAULTS['chat']}\n  ollama pull {LOCAL_DEFAULTS['code']}")
    out("Check with: turbo doctor")
    return 0


def _restrict_permissions(path) -> None:
    """Make the settings file readable only by this user (where supported)."""
    if os.name != "nt":
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass


__all__ = ["run_setup", "usable_models", "provider_label"]
