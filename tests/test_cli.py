import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

import turbo
from tests.helpers import make_assistant, temp_dir
from tests.test_assistant import FIX_REPLY
from turbo_core.llm import LLMConnectionError

FILE_TEXT = (
    "import math\r\n"
    "\r\n"
    "def average(xs):\r\n"
    "    return sum(xs) / len(xs) + 1\r\n"
    "\r\n"
    "print(average([1, 2]))\r\n"
)


class CliTests(unittest.TestCase):
    def setUp(self):
        self.dir = temp_dir(self)
        self.file = self.dir / "stats.py"
        self.file.write_bytes(FILE_TEXT.encode())

    def run_cli(self, argv, *replies, stdin=""):
        assistant, llm = make_assistant(self, *replies)
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err), mock.patch("sys.stdin", io.StringIO(stdin)):
            code = turbo.main(argv, assistant=assistant)
        return code, out.getvalue(), err.getvalue(), llm

    def test_parse_lines(self):
        self.assertEqual(turbo.parse_lines(None, 10), (1, 10))
        self.assertEqual(turbo.parse_lines("3-5", 10), (3, 5))
        self.assertEqual(turbo.parse_lines("7", 10), (7, 7))
        self.assertEqual(turbo.parse_lines("8-99", 10), (8, 10))
        for bad in ("0", "5-3", "11", "a-b"):
            with self.assertRaises(turbo.InputError):
                turbo.parse_lines(bad, 10)

    def test_fix_prints_diff_with_real_line_numbers(self):
        code, out, _, llm = self.run_cli(["fix", str(self.file), "-l", "3-4", "--no-color"], FIX_REPLY)
        self.assertEqual(code, 0)
        self.assertIn("one too high", out)
        self.assertIn("@@ -3,2 +3,2 @@", out)
        self.assertIn("+    return sum(xs) / len(xs)", out)
        # Surrounding lines were sent as context.
        self.assertIn("print(average", llm.calls[0]["messages"][-1]["content"])
        self.assertEqual(self.file.read_bytes().decode(), FILE_TEXT)  # untouched

    def test_fix_apply_yes_writes_file_and_keeps_crlf(self):
        code, out, _, _ = self.run_cli(["fix", str(self.file), "-l", "3-4", "--apply", "-y"], FIX_REPLY)
        self.assertEqual(code, 0)
        self.assertEqual(
            self.file.read_bytes().decode(),
            FILE_TEXT.replace("len(xs) + 1", "len(xs)"),
        )

    def test_fix_apply_declined(self):
        with mock.patch("builtins.input", return_value="n"):
            code, out, _, _ = self.run_cli(["fix", str(self.file), "-l", "3-4", "--apply"], FIX_REPLY)
        self.assertIn("Not applied", out)
        self.assertEqual(self.file.read_bytes().decode(), FILE_TEXT)

    def test_fix_from_stdin(self):
        code, out, _, _ = self.run_cli(
            ["fix", "-", "--no-color"], FIX_REPLY,
            stdin="def average(xs):\n    return sum(xs) / len(xs) + 1\n",
        )
        self.assertEqual(code, 0)
        self.assertIn("--- a/stdin", out)

    def test_explain(self):
        code, out, _, llm = self.run_cli(["explain", str(self.file), "-q", "is it fast?"], "Yes.")
        self.assertEqual(code, 0)
        self.assertEqual(out.strip(), "Yes.")
        self.assertIn("is it fast?", llm.calls[0]["messages"][-1]["content"])

    def test_ask_with_file_and_session(self):
        code, out, _, llm = self.run_cli(["-s", "proj", "ask", "what", "is", "this?", "-f", str(self.file)], "Stats.")
        self.assertEqual(out.strip(), "Stats.")
        self.assertIn("what is this?", llm.calls[0]["messages"][-1]["content"])

    def test_missing_file_exit_code_2(self):
        code, _, err, _ = self.run_cli(["fix", str(self.dir / "nope.py")])
        self.assertEqual(code, 2)
        self.assertIn("File not found", err)

    def test_ollama_down_exit_code_1(self):
        code, _, err, _ = self.run_cli(["ask", "hi"], LLMConnectionError("Can't reach Ollama"))
        self.assertEqual(code, 1)
        self.assertIn("Can't reach Ollama", err)

    def test_fix_streams_explanation_but_not_code(self):
        code, out, _, llm = self.run_cli(["fix", str(self.file), "-l", "3-4", "--no-color"], FIX_REPLY)
        self.assertTrue(llm.calls[0]["stream"])
        self.assertNotIn("EXPLANATION", out)
        self.assertNotIn("FIXED CODE", out)
        self.assertNotIn("```", out)
        self.assertEqual(out.count("one too high"), 1)  # streamed once, not repeated
        self.assertIn("@@ -3,2 +3,2 @@", out)

    def test_no_stream_flag(self):
        code, out, _, llm = self.run_cli(["--no-stream", "ask", "hi"], "Hello!")
        self.assertFalse(llm.calls[0]["stream"])
        self.assertEqual(out.strip(), "Hello!")

    def test_no_command_opens_chat(self):
        code, out, _, llm = self.run_cli([], "hello back", stdin="hello\n/exit\n")
        self.assertEqual(code, 0)
        self.assertIn("TURBO chat", out)
        self.assertIn("hello back", out)

    def test_chat_follow_up_and_exit(self):
        code, out, _, llm = self.run_cli(["chat"], "A1", "A2", stdin="q1\nq2\n/exit\n")
        self.assertEqual(code, 0)
        self.assertEqual([m["content"] for m in llm.calls[1]["messages"][1:]], ["q1", "A1", "q2"])


if __name__ == "__main__":
    unittest.main()


class FixExplanationFilterTests(unittest.TestCase):
    def run_filter(self, reply, chunk_size):
        out = []
        f = turbo.FixExplanationFilter(out.append)
        for i in range(0, len(reply), chunk_size):
            f(reply[i:i + chunk_size])
        f.finish()
        return "".join(out)

    def test_every_chunk_size_gives_clean_explanation(self):
        reply = "EXPLANATION:\nThe + 1 is wrong.\n\nFIXED CODE:\n```python\nx = 1\n```"
        for size in range(1, 20):
            self.assertEqual(self.run_filter(reply, size), "The + 1 is wrong.", size)

    def test_lowercase_markers(self):
        self.assertEqual(self.run_filter("explanation: bad\nfixed code:\n```\nx\n```", 2), "bad")

    def test_reply_without_markers_passes_through(self):
        self.assertEqual(self.run_filter("No bug here, it's fine.", 4), "No bug here, it's fine.")

    def test_short_reply(self):
        self.assertEqual(self.run_filter("Ok.", 1), "Ok.")
