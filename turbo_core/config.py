"""Settings loaded from environment variables (and an optional .env file).

Nothing secret is hardcoded. Every setting has a safe local default, so TURBO
runs with no configuration as long as Ollama is running on this machine.

Per-user files live in the TURBO home folder: ~/.turbo by default, or the
folder named by TURBO_HOME. It holds turbo.db (conversation history) and an
optional .env. A .env in the current folder is deliberately NOT read, so a
cloned repository can't silently redirect TURBO.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def turbo_home() -> Path:
    """Folder for per-user TURBO files (~/.turbo unless TURBO_HOME is set)."""
    custom = os.environ.get("TURBO_HOME", "").strip()
    return Path(custom).expanduser() if custom else Path.home() / ".turbo"


def load_dotenv(path: Path | str | None = None) -> None:
    """Load KEY=VALUE lines from a .env file into os.environ.

    Real environment variables always win over the file. Lines starting with
    '#' and blank lines are ignored; surrounding quotes are stripped.
    """
    path = Path(path) if path else turbo_home() / ".env"
    if not path.is_file():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value


def _int_env(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a whole number, got {value!r}") from error


@dataclass(frozen=True)
class Settings:
    ollama_url: str = "http://localhost:11434"
    # Small, fast defaults; switch with /model or TURBO_*_MODEL.
    chat_model: str = "llama3.2:3b"
    code_model: str = "qwen2.5-coder:3b"
    timeout_seconds: int = 300
    # Starting context window; grows up to max_context_tokens for big files.
    context_tokens: int = 4096
    max_context_tokens: int = 16384
    # Answer length cap for questions/explanations (fixes are not capped,
    # because a cut-off fix can't be turned into a diff).
    max_tokens: int = 1200
    # How long Ollama keeps the model loaded after a request.
    keep_alive: str = "30m"
    # Earlier conversation sent with each request. Less = faster on CPU.
    history_chars: int = 6000
    db_path: Path = field(default_factory=lambda: turbo_home() / "turbo.db")
    host: str = "127.0.0.1"
    port: int = 8765
    token: str = field(default="", repr=False)
    # Where the model runs: "ollama" (this computer) or a cloud provider
    # with a free tier: "groq", "gemini", "openrouter". Set up with `turbo setup`.
    provider: str = "ollama"
    api_key: str = field(default="", repr=False)  # never printed
    api_base: str = ""  # optional override of the provider's API URL
    # Conversation history is kept in memory for the session only, unless
    # this is turned on (TURBO_SAVE_HISTORY=1), which saves it to db_path.
    save_history: bool = False

    @property
    def is_cloud(self) -> bool:
        return self.provider != "ollama"

    @classmethod
    def from_env(cls, dotenv: bool = True) -> "Settings":
        if dotenv:
            load_dotenv()
        defaults = cls()
        db_path = os.environ.get("TURBO_DB_PATH", "").strip()
        provider = os.environ.get("TURBO_PROVIDER", defaults.provider).strip().lower() or "ollama"
        return cls(
            ollama_url=os.environ.get("TURBO_OLLAMA_URL", defaults.ollama_url).rstrip("/"),
            chat_model=os.environ.get("TURBO_CHAT_MODEL", defaults.chat_model),
            code_model=os.environ.get("TURBO_CODE_MODEL", defaults.code_model),
            timeout_seconds=_int_env("TURBO_TIMEOUT", defaults.timeout_seconds),
            context_tokens=_int_env("TURBO_CONTEXT_TOKENS", defaults.context_tokens),
            max_context_tokens=_int_env("TURBO_MAX_CONTEXT_TOKENS", defaults.max_context_tokens),
            max_tokens=_int_env("TURBO_MAX_TOKENS", defaults.max_tokens),
            keep_alive=os.environ.get("TURBO_KEEP_ALIVE", defaults.keep_alive),
            history_chars=_int_env("TURBO_HISTORY_CHARS", defaults.history_chars),
            db_path=Path(db_path) if db_path else defaults.db_path,
            host=os.environ.get("TURBO_HOST", defaults.host),
            port=_int_env("TURBO_PORT", defaults.port),
            token=os.environ.get("TURBO_TOKEN", defaults.token),
            provider=provider,
            api_key=_api_key(provider),
            api_base=os.environ.get("TURBO_API_BASE", "").strip(),
            save_history=os.environ.get("TURBO_SAVE_HISTORY", "").strip().lower() in ("1", "true", "yes", "on"),
        )


def _api_key(provider: str) -> str:
    """TURBO_API_KEY, or the provider's usual variable (e.g. GROQ_API_KEY)."""
    key = os.environ.get("TURBO_API_KEY", "").strip()
    if key or provider == "ollama":
        return key
    from .cloud import PROVIDERS

    env_name = PROVIDERS.get(provider, {}).get("env_key")
    return os.environ.get(env_name, "").strip() if env_name else ""


def save_setting(key: str, value: str, path: Path | None = None) -> Path:
    """Set KEY=value in ~/.turbo/.env (creating it), keeping other lines.

    Also updates os.environ so the new value applies right away.
    """
    if not key.startswith("TURBO_") or not key.replace("_", "").isalnum():
        raise ValueError(f"Not a TURBO setting: {key!r}")
    if "\n" in value or "\r" in value:
        raise ValueError("Setting values can't contain line breaks.")
    path = Path(path) if path else turbo_home() / ".env"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
    new_line = f"{key}={value}"
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("export "):
            stripped = stripped[len("export "):]
        if stripped.split("=", 1)[0].strip() == key:
            lines[i] = new_line
            break
    else:
        lines.append(new_line)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.environ[key] = value
    return path
