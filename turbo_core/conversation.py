"""Per-session conversation history stored in SQLite."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

TRUNCATED_NOTE = "[... earlier part of this message truncated ...]\n"


class ConversationStore:
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
        """Most recent messages that fit in max_chars, oldest first.

        If even the newest message is too big it is kept, cut down from the
        front, so a follow-up still sees the end of what was just discussed.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT role, content FROM messages WHERE session_id = ? "
                "ORDER BY id DESC LIMIT 200",
                (session_id,),
            ).fetchall()

        picked: list[dict[str, str]] = []
        used = 0
        for role, content in rows:
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

    def clear(self, session_id: str) -> int:
        with self._connect() as conn:
            return conn.execute(
                "DELETE FROM messages WHERE session_id = ?", (session_id,)
            ).rowcount
