"""Speed-related behavior: context sizing, answer caps, model switching, settings."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests.helpers import make_assistant
from tests.test_assistant import BUGGY, FIX_REPLY
from turbo_core.config import save_setting
from turbo_core.llm import LLMInputTooLarge, OllamaClient


def msgs(chars):
    return [{"role": "user", "content": "x" * chars}]


class ContextSizingTests(unittest.TestCase):
    def test_small_prompt_keeps_default(self):
        client = OllamaClient(context_tokens=4096, max_context_tokens=16384)
        self.assertEqual(client.context_for(msgs(1000)), 4096)

    def test_grows_in_steps_and_never_shrinks(self):
        client = OllamaClient(context_tokens=4096, max_context_tokens=16384)
        grown = client.context_for(msgs(20_000))  # ~6.7k tokens + margin
        self.assertEqual(grown % 2048, 0)
        self.assertGreater(grown, 4096)
        # A later small request keeps the bigger window, so Ollama doesn't reload.
        self.assertEqual(client.context_for(msgs(100)), grown)

    def test_expected_output_counts(self):
        client = OllamaClient(context_tokens=4096, max_context_tokens=16384)
        self.assertGreater(client.context_for(msgs(6000), expected_output_chars=9000), 4096)

    def test_too_large_is_a_clear_error(self):
        client = OllamaClient(context_tokens=4096, max_context_tokens=8192)
        with self.assertRaisesRegex(LLMInputTooLarge, "/clear"):
            client.context_for(msgs(100_000))

    def test_payload_has_keep_alive_and_num_predict(self):
        session = mock.Mock()
        session.post.return_value = mock.Mock(status_code=200, json=lambda: {"message": {"content": "ok"}})
        client = OllamaClient(session=session, keep_alive="45m")
        client.chat([{"role": "user", "content": "hi"}], model="m", max_tokens=100)
        payload = session.post.call_args.kwargs["json"]
        self.assertEqual(payload["keep_alive"], "45m")
        self.assertEqual(payload["options"]["num_predict"], 100)
        client.chat([{"role": "user", "content": "hi"}], model="m")
        self.assertEqual(session.post.call_args.kwargs["json"]["options"]["num_predict"], -1)


class AnswerCapTests(unittest.TestCase):
    def test_questions_are_capped_fixes_are_not(self):
        assistant, llm = make_assistant(self, "answer", FIX_REPLY, max_tokens=321)
        assistant.ask("q")
        assistant.fix(BUGGY)
        self.assertEqual(llm.calls[0]["max_tokens"], 321)
        self.assertIsNone(llm.calls[1]["max_tokens"])
        self.assertGreater(llm.calls[1]["expected_output_chars"], len(BUGGY))

    def test_truncation_is_reported(self):
        assistant, llm = make_assistant(self, "partial", "full")
        llm.next_done_reason = "length"
        assistant.ask("q")
        self.assertTrue(assistant.last_reply_truncated)
        llm.next_done_reason = "stop"
        assistant.ask("q")
        self.assertFalse(assistant.last_reply_truncated)


class ModelSwitchTests(unittest.TestCase):
    def test_set_models(self):
        assistant, llm = make_assistant(self, "a", "b")
        assistant.set_models(chat="fast:1b")
        assistant.ask("hi")
        self.assertEqual(llm.calls[0]["model"], "fast:1b")
        assistant.set_models(code="coder:1b")
        assistant.explain("x = 1")
        self.assertEqual(llm.calls[1]["model"], "coder:1b")
        self.assertEqual(assistant.settings.chat_model, "fast:1b")


class SaveSettingTests(unittest.TestCase):
    def setUp(self):
        self.path = Path(tempfile.mkdtemp()) / "sub" / ".env"
        self.addCleanup(lambda: __import__("shutil").rmtree(self.path.parent.parent, ignore_errors=True))

    def test_creates_updates_and_keeps_other_lines(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            save_setting("TURBO_CODE_MODEL", "a:1", self.path)
            self.path.write_text("# my notes\nTURBO_PORT=9000\n" + self.path.read_text(), encoding="utf-8")
            save_setting("TURBO_CODE_MODEL", "b:2", self.path)
            self.assertEqual(os.environ["TURBO_CODE_MODEL"], "b:2")
        text = self.path.read_text(encoding="utf-8")
        self.assertIn("# my notes", text)
        self.assertIn("TURBO_PORT=9000", text)
        self.assertIn("TURBO_CODE_MODEL=b:2", text)
        self.assertNotIn("a:1", text)

    def test_rejects_non_turbo_keys_and_newlines(self):
        with self.assertRaises(ValueError):
            save_setting("PATH", "x", self.path)
        with self.assertRaises(ValueError):
            save_setting("TURBO_X", "a\nTURBO_TOKEN=evil", self.path)


if __name__ == "__main__":
    unittest.main()
