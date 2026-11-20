import unittest

from turbo_core.assistant import InputError
from turbo_core.llm import LLMConnectionError
from turbo_core.prompts import SYSTEM_PROMPT
from tests.helpers import make_assistant

BUGGY = "def average(xs):\n    return sum(xs) / len(xs) + 1\n"
FIX_REPLY = (
    "EXPLANATION:\nThe `+ 1` makes every average one too high. I removed it.\n\n"
    "FIXED CODE:\n```python\ndef average(xs):\n    return sum(xs) / len(xs)\n```"
)


class AskTests(unittest.TestCase):
    def test_follow_up_sees_earlier_messages(self):
        assistant, llm = make_assistant(self, "Paris.", "About 2.1 million.")
        assistant.ask("Capital of France?", session_id="s")
        assistant.ask("What's its population?", session_id="s")

        sent = llm.calls[1]["messages"]
        self.assertEqual(sent[0], {"role": "system", "content": SYSTEM_PROMPT})
        self.assertEqual(
            [m["content"] for m in sent[1:]],
            ["Capital of France?", "Paris.", "What's its population?"],
        )

    def test_uses_chat_model_without_code_and_code_model_with_code(self):
        assistant, llm = make_assistant(self, "a", "b")
        assistant.ask("hi")
        assistant.ask("what does this do?", context="x = 1", language="python")
        self.assertEqual(llm.calls[0]["model"], "chat-model")
        self.assertEqual(llm.calls[1]["model"], "code-model")
        self.assertIn("```python\nx = 1\n```", llm.calls[1]["messages"][-1]["content"])

    def test_failed_call_is_not_saved_to_history(self):
        assistant, llm = make_assistant(self, LLMConnectionError("down"), "ok")
        with self.assertRaises(LLMConnectionError):
            assistant.ask("first", session_id="s")
        assistant.ask("second", session_id="s")
        self.assertEqual(len(llm.calls[1]["messages"]), 2)  # system + "second"

    def test_empty_question_rejected(self):
        assistant, _ = make_assistant(self)
        with self.assertRaises(InputError):
            assistant.ask("   ")

    def test_reset_clears_history(self):
        assistant, llm = make_assistant(self, "a", "b")
        assistant.ask("q1", session_id="s")
        assistant.reset("s")
        assistant.ask("q2", session_id="s")
        self.assertEqual(len(llm.calls[1]["messages"]), 2)


class ExplainTests(unittest.TestCase):
    def test_prompt_includes_code_filename_and_question(self):
        assistant, llm = make_assistant(self, "It averages numbers.")
        answer = assistant.explain(BUGGY, filename="stats.py", question="Is this safe?")
        self.assertEqual(answer, "It averages numbers.")
        prompt = llm.calls[0]["messages"][-1]["content"]
        self.assertIn("Is this safe?", prompt)
        self.assertIn("`stats.py` (python)", prompt)
        self.assertIn("return sum(xs)", prompt)
        self.assertEqual(llm.calls[0]["model"], "code-model")

    def test_rejects_huge_input(self):
        assistant, _ = make_assistant(self)
        with self.assertRaisesRegex(InputError, "limit"):
            assistant.explain("x" * 100_000)


class FixTests(unittest.TestCase):
    def test_returns_explanation_and_unified_diff(self):
        assistant, _ = make_assistant(self, FIX_REPLY)
        result = assistant.fix(BUGGY, filename="stats.py")
        self.assertTrue(result.changed)
        self.assertIn("one too high", result.explanation)
        self.assertEqual(result.fixed_code, "def average(xs):\n    return sum(xs) / len(xs)\n")
        self.assertIn("-    return sum(xs) / len(xs) + 1\n+    return sum(xs) / len(xs)\n", result.diff)
        self.assertIn("--- a/stats.py", result.diff)

    def test_indented_crlf_selection_keeps_its_shape(self):
        selection = "    def average(xs):\r\n        return sum(xs) / len(xs) + 1\r\n"
        assistant, llm = make_assistant(self, FIX_REPLY)
        result = assistant.fix(selection, filename="stats.py", start_line=10)
        self.assertEqual(
            result.fixed_code, "    def average(xs):\r\n        return sum(xs) / len(xs)\r\n"
        )
        self.assertIn("@@ -10,2 +10,2 @@", result.diff)
        # Model saw dedented LF code.
        self.assertIn("```python\ndef average(xs):\n", llm.calls[0]["messages"][-1]["content"])

    def test_no_code_in_reply_changes_nothing(self):
        assistant, _ = make_assistant(self, "I couldn't find a bug.")
        result = assistant.fix(BUGGY)
        self.assertFalse(result.changed)
        self.assertEqual(result.fixed_code, BUGGY)
        self.assertEqual(result.diff, "")
        self.assertTrue(result.warnings)

    def test_unchanged_code_means_no_diff(self):
        reply = "EXPLANATION:\nNo bug.\nFIXED CODE:\n```python\n" + BUGGY + "```"
        assistant, _ = make_assistant(self, reply)
        result = assistant.fix(BUGGY)
        self.assertFalse(result.changed)
        self.assertEqual(result.diff, "")

    def test_warns_when_model_returns_only_a_fragment(self):
        long_code = "".join(f"line_{i} = {i}\n" for i in range(20))
        reply = "EXPLANATION:\nx\nFIXED CODE:\n```\nline_3 = 4\n```"
        assistant, _ = make_assistant(self, reply)
        result = assistant.fix(long_code)
        self.assertTrue(any("shorter" in w for w in result.warnings))

    def test_follow_up_after_fix_has_context(self):
        assistant, llm = make_assistant(self, FIX_REPLY, "Because of the + 1.")
        assistant.fix(BUGGY, session_id="s")
        assistant.ask("Why was it wrong?", session_id="s")
        history = llm.calls[1]["messages"]
        self.assertIn("return sum(xs) / len(xs) + 1", history[1]["content"])
        self.assertEqual(history[2]["content"], FIX_REPLY)
        # Follow-up about code goes to the code model, not the chat model.
        self.assertEqual(llm.calls[1]["model"], "code-model")


if __name__ == "__main__":
    unittest.main()
