import unittest

from turbo_core import diffing
from turbo_core.prompts import fence


class ParseFixReplyTests(unittest.TestCase):
    def test_standard_format(self):
        reply = (
            "EXPLANATION:\nOff by one in the loop.\n\nFIXED CODE:\n"
            "```python\nfor i in range(n):\n    print(i)\n```\n"
        )
        parsed = diffing.parse_fix_reply(reply)
        self.assertEqual(parsed.explanation, "Off by one in the loop.")
        self.assertEqual(parsed.code, "for i in range(n):\n    print(i)")

    def test_code_in_explanation_is_ignored(self):
        reply = (
            "EXPLANATION:\nThe line ```x = 1``` is wrong; e.g.\n```python\nbad()\n```\n"
            "FIXED CODE:\n```python\ngood()\n```"
        )
        self.assertEqual(diffing.parse_fix_reply(reply).code, "good()")

    def test_no_marker_uses_last_block(self):
        reply = "Bug here:\n```py\nold()\n```\nFixed:\n```py\nnew()\n```"
        parsed = diffing.parse_fix_reply(reply)
        self.assertEqual(parsed.code, "new()")
        self.assertTrue(parsed.explanation.startswith("Bug here"))

    def test_no_code_block(self):
        parsed = diffing.parse_fix_reply("I don't see a bug.")
        self.assertIsNone(parsed.code)
        self.assertEqual(parsed.explanation, "I don't see a bug.")

    def test_long_fence_with_backticks_inside(self):
        code = 'doc = """\n```\nexample\n```\n"""'
        reply = "EXPLANATION:\nx\nFIXED CODE:\n" + fence(code, "python")
        self.assertEqual(diffing.parse_fix_reply(reply).code, code)


class ShapeTests(unittest.TestCase):
    def test_roundtrip_crlf_indent_and_trailing_newline(self):
        original = "    def f():\r\n        return 1\r\n"
        shape = diffing.analyze(original)
        self.assertEqual(shape.eol, "\r\n")
        self.assertEqual(shape.margin, "    ")
        self.assertTrue(shape.trailing_newline)

        sent = diffing.dedent(original, shape)
        self.assertEqual(sent, "def f():\n    return 1\n")

        restored = diffing.restore("def f():\n    return 2", shape)
        self.assertEqual(restored, "    def f():\r\n        return 2\r\n")

    def test_restore_does_not_double_indent(self):
        shape = diffing.analyze("    x = 1\n    y = 2")
        self.assertEqual(diffing.restore("    x = 1\n    y = 3\n", shape), "    x = 1\n    y = 3")

    def test_blank_lines_stay_blank(self):
        shape = diffing.analyze("  a\n\n  b")
        self.assertEqual(diffing.restore("a\n\nc", shape), "  a\n\n  c")


class UnifiedDiffTests(unittest.TestCase):
    def test_basic_diff(self):
        diff = diffing.unified_diff("a\nb\nc\n", "a\nB\nc\n", "src/x.py")
        self.assertIn("--- a/src/x.py", diff)
        self.assertIn("+++ b/src/x.py", diff)
        self.assertIn("-b\n+B\n", diff)

    def test_identical_is_empty(self):
        self.assertEqual(diffing.unified_diff("a\n", "a\n"), "")

    def test_crlf_only_difference_is_ignored(self):
        self.assertEqual(diffing.unified_diff("a\r\nb\r\n", "a\nb\n"), "")

    def test_start_line_offsets_hunks(self):
        diff = diffing.unified_diff("a\nb\n", "a\nc\n", "f.py", start_line=41)
        self.assertIn("@@ -41,2 +41,2 @@", diff)

    def test_missing_final_newline_marker(self):
        diff = diffing.unified_diff("a", "b")
        self.assertIn("\\ No newline at end of file", diff)

    def test_windows_paths_use_forward_slashes(self):
        self.assertIn("a/src/x.py", diffing.unified_diff("a\n", "b\n", "src\\x.py"))


if __name__ == "__main__":
    unittest.main()
