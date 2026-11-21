import shutil
import tempfile
from pathlib import Path

from turbo_core.assistant import Assistant
from turbo_core.config import Settings
from turbo_core.conversation import ConversationStore


class FakeLLM:
    """Records calls and returns queued replies."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []
        self.last_done_reason = None
        self.next_done_reason = "stop"  # set to "length" to fake a cut-off answer

    def chat(self, messages, model, temperature=0.2, **options):
        self.calls.append({"messages": messages, "model": model, "stream": False, **options})
        reply = self._next()
        self.last_done_reason = self.next_done_reason
        return reply

    def chat_stream(self, messages, model, temperature=0.2, chunk_size=3, **options):
        self.calls.append({"messages": messages, "model": model, "stream": True, **options})
        reply = self._next()
        self.last_done_reason = self.next_done_reason
        # Small chunks so markers get split across pieces, like real streaming.
        for i in range(0, len(reply), chunk_size):
            yield reply[i:i + chunk_size]

    def _next(self):
        if not self.replies:
            raise AssertionError("FakeLLM ran out of replies")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def temp_dir(test_case) -> Path:
    path = Path(tempfile.mkdtemp(prefix="turbo-test-"))
    test_case.addCleanup(shutil.rmtree, path, ignore_errors=True)
    return path


def make_assistant(test_case, *replies, **settings_overrides):
    settings = Settings(
        db_path=temp_dir(test_case) / "test.db",
        chat_model="chat-model",
        code_model="code-model",
        **settings_overrides,
    )
    llm = FakeLLM(*replies)
    return Assistant(llm, ConversationStore(settings.db_path), settings), llm
