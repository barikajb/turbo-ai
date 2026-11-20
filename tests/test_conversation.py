import unittest

from turbo_core.conversation import TRUNCATED_NOTE, ConversationStore
from tests.helpers import temp_dir


class ConversationStoreTests(unittest.TestCase):
    def setUp(self):
        self.store = ConversationStore(temp_dir(self) / "sub" / "c.db")

    def test_history_in_order(self):
        self.store.add_exchange("s", "q1", "a1")
        self.store.add_exchange("s", "q2", "a2")
        self.assertEqual(
            [m["content"] for m in self.store.history("s", 1000)], ["q1", "a1", "q2", "a2"]
        )

    def test_sessions_are_separate(self):
        self.store.add_exchange("one", "q", "a")
        self.assertEqual(self.store.history("two", 1000), [])

    def test_budget_keeps_newest_and_starts_with_user(self):
        self.store.add_exchange("s", "x" * 50, "y" * 50)
        self.store.add_exchange("s", "q2", "a2")
        # 80 chars: fits q2+a2 and "y"*50 but not the first question, and
        # a lone assistant message must not lead the history.
        history = self.store.history("s", 80)
        self.assertEqual([m["content"] for m in history], ["q2", "a2"])

    def test_oversized_reply_is_truncated_and_keeps_its_question(self):
        self.store.add_exchange("s", "why?", "z" * 1000)
        history = self.store.history("s", 300)
        self.assertEqual([m["role"] for m in history], ["user", "assistant"])
        self.assertEqual(history[0]["content"], "why?")
        self.assertTrue(history[1]["content"].startswith(TRUNCATED_NOTE))
        self.assertLessEqual(sum(len(m["content"]) for m in history), 300)

    def test_oversized_question_is_truncated_from_the_front(self):
        self.store.add_exchange("s", "Q" * 1000 + "END", "a")
        history = self.store.history("s", 300)
        self.assertEqual([m["role"] for m in history], ["user", "assistant"])
        self.assertTrue(history[0]["content"].startswith(TRUNCATED_NOTE))
        self.assertTrue(history[0]["content"].endswith("END"))

    def test_clear(self):
        self.store.add_exchange("s", "q", "a")
        self.assertEqual(self.store.clear("s"), 2)
        self.assertEqual(self.store.history("s", 1000), [])

    def test_persists_across_instances(self):
        self.store.add_exchange("s", "q", "a")
        again = ConversationStore(self.store.db_path)
        self.assertEqual(len(again.history("s", 1000)), 2)


if __name__ == "__main__":
    unittest.main()
