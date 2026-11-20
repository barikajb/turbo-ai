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
    chat_model: str = "llama3.2:3b"
    code_model: str = "qwen2.5-coder:7b"
    timeout_seconds: int = 300
    context_tokens: int = 8192
    history_chars: int = 12000
    db_path: Path = field(default_factory=lambda: turbo_home() / "turbo.db")
    host: str = "127.0.0.1"
    port: int = 8765
    token: str = ""

    @classmethod
    def from_env(cls, dotenv: bool = True) -> "Settings":
        if dotenv:
            load_dotenv()
        defaults = cls()
        db_path = os.environ.get("TURBO_DB_PATH", "").strip()
        return cls(
            ollama_url=os.environ.get("TURBO_OLLAMA_URL", defaults.ollama_url).rstrip("/"),
            chat_model=os.environ.get("TURBO_CHAT_MODEL", defaults.chat_model),
            code_model=os.environ.get("TURBO_CODE_MODEL", defaults.code_model),
            timeout_seconds=_int_env("TURBO_TIMEOUT", defaults.timeout_seconds),
            context_tokens=_int_env("TURBO_CONTEXT_TOKENS", defaults.context_tokens),
            history_chars=_int_env("TURBO_HISTORY_CHARS", defaults.history_chars),
            db_path=Path(db_path) if db_path else defaults.db_path,
            host=os.environ.get("TURBO_HOST", defaults.host),
            port=_int_env("TURBO_PORT", defaults.port),
            token=os.environ.get("TURBO_TOKEN", defaults.token),
        )
