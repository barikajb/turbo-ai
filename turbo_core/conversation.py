"""Conversation history for follow-up questions.

By default history lives only in memory (RAM) for the current session: it is
never written to disk and disappears when TURBO exits or the terminal closes.
Set TURBO_SAVE_HISTORY=1 to keep it on disk (SQLite) between sessions instead.
"""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

TRUNCATED_NOTE = "[... earlier part of this message truncated ...]\n"
MAX_MESSAGES = 200  # per session, newest kept


def select_history(newest_first: list[tuple[str, str]], max_chars: int) -> list[dict[str, str]]:
    """Most recent messages that fit in max_chars, oldest first.

    If even the newest message is too big it is kept, cut down from the
    front, so a follow-up still sees the end of what was just discussed.
    """
    picked: list[dict[str, str]] = []
    used = 0
    for role, content in newest_first:
        remaining = max_chars - used
        if len(content) <= remaining:
            picked.append({"role": role, "content": content})
            used += len(content)
            continue
        # Too big. Still keep a truncated copy if it belongs to the latest
        # exchange; leave a third of the budget for the question before a
        # huge reply.
        if len(picked) < 2:
            allowance = max_chars * 2 // 3 if not picked and role == "assistant" else remaining
            keep = allowance - len(TRUNCATED_NOTE)
            if keep > 50:
                picked.append({"role": role, "content": TRUNCATED_NOTE + content[-keep:]})
                used += allowance
                continue
        break

    picked.reverse()
    # Chat models expect history to start with the user.
    while picked and picked[0]["role"] != "user":
        picked.pop(0)
    return picked


class MemoryConversationStore:
    """Session-only history. Nothing is saved; it's gone when TURBO exits."""

    persistent = False

    def __init__(self):
        self._sessions: dict[str, list[tuple[str, str]]] = {}
        self._lock = threading.Lock()  # the VS Code server uses threads

    def add_exchange(self, session_id: str, user: str, assistant: str) -> None:
        with self._lock:
            messages = self._sessions.setdefault(session_id, [])
            messages.extend([("user", user), ("assistant", assistant)])
            del messages[:-MAX_MESSAGES]

    def history(self, session_id: str, max_chars: int) -> list[dict[str, str]]:
        with self._lock:
            newest_first = list(reversed(self._sessions.get(session_id, [])))
        return select_history(newest_first, max_chars)

    def clear(self, session_id: str) -> int:
        with self._lock:
            return len(self._sessions.pop(session_id, []))


class ConversationStore:
    """History saved on disk in SQLite (only when TURBO_SAVE_HISTORY=1)."""

    persistent = True

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_session ON messages (session_id, id)"
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        # A fresh connection per call keeps this safe across server threads.
        conn = sqlite3.connect(self.db_path, timeout=10)
        try:
            with conn:  # commit on success, roll back on error
                yield conn
        finally:
            conn.close()

    def add_exchange(self, session_id: str, user: str, assistant: str) -> None:
        with self._connect() as conn:
            conn.executemany(
                "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
                [(session_id, "user", user), (session_id, "assistant", assistant)],
            )

    def history(self, session_id: str, max_chars: int) -> list[dict[str, str]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT role, content FROM messages WHERE session_id = ? "
                "ORDER BY id DESC LIMIT ?",
                (session_id, MAX_MESSAGES),
            ).fetchall()
        return select_history(rows, max_chars)

    def clear(self, session_id: str) -> int:
        with self._connect() as conn:
            return conn.execute(
                "DELETE FROM messages WHERE session_id = ?", (session_id,)
            ).rowcount
