import io
import unittest
from unittest import mock

from rich.console import Console

from tests.helpers import make_assistant, temp_dir
from turbo_core import system, ui
from turbo_core.llm import LLMConnectionError

GB = system.GB


def console():
    return Console(file=io.StringIO(), force_terminal=False, width=100, height=40)


def output(c):
    return c.file.getvalue()


def installed(*names):
    return [{"name": n, "size": 2 * GB, "parameters": "3.1B", "quantization": "Q4"} for n in names]


class StreamCallTests(unittest.TestCase):
    def test_renders_full_answer_and_returns_result(self):
        c = console()

        def call(on_token):
            for piece in ["# Title\n", "Some `code` ", "here."]:
                on_token(piece)
            return "RESULT"

        result, text = ui.stream_call(c, call)
        self.assertEqual(result, "RESULT")
        self.assertEqual(text, "# Title\nSome `code` here.")
        self.assertIn("Title", output(c))
        self.assertIn("code here.", output(c))

    def test_errors_are_reraised(self):
        def call(on_token):
            raise LLMConnectionError("down")

        with self.assertRaises(LLMConnectionError):
            ui.stream_call(console(), call)

    def test_ctrl_c_stops_and_reports(self):
        c = console()
        with mock.patch.object(ui.queue.Queue, "get", side_effect=KeyboardInterrupt):
            result, _ = ui.stream_call(c, lambda on_token: "never")
        self.assertIsNone(result)
        self.assertIn("(stopped)", output(c))


class ChatCommandTests(unittest.TestCase):
    def make_ui(self, *replies):
        assistant, llm = make_assistant(self, *replies)
        llm.list_models = lambda: ["chat-model", "code-model", "small:1b"]
        llm.model_details = lambda: installed("chat-model", "code-model", "small:1b")
        llm.loaded_models = lambda: []
        c = console()
        chat = ui.ChatUI(assistant, "s", console=c)
        chat.warm = lambda model: None
        return chat, assistant, llm, c

    def test_help_lists_commands_and_keys(self):
        chat, _, _, c = self.make_ui()
        chat.handle_command("/help")
        for text in ("/model", "/manage", "/clear", "Ctrl+L", "Alt+Enter"):
            self.assertIn(text, output(c))

    def test_model_switch_both_and_one_role(self):
        chat, assistant, _, _ = self.make_ui()
        chat.handle_command("/model small:1b")
        self.assertEqual(assistant.settings.chat_model, "small:1b")
        self.assertEqual(assistant.settings.code_model, "small:1b")
        chat.handle_command("/model code code-model")
        self.assertEqual(assistant.settings.code_model, "code-model")
        self.assertEqual(assistant.settings.chat_model, "small:1b")

    def test_model_not_installed_declined(self):
        chat, assistant, _, c = self.make_ui()
        chat.confirm = lambda question, default=False: False
        chat.handle_command("/model missing:7b")
        self.assertEqual(assistant.settings.code_model, "code-model")
        self.assertIn("Not switched", output(c))

    def test_model_not_installed_pulls_when_accepted(self):
        chat, assistant, llm, _ = self.make_ui()
        chat.confirm = lambda question, default=False: True
        pulled = []
        llm.pull = lambda name, on_progress=None: pulled.append(name)
        chat.handle_command("/model chat new:1b")
        self.assertEqual(pulled, ["new:1b"])
        self.assertEqual(assistant.settings.chat_model, "new:1b")

    def test_model_save_persists(self):
        chat, _, _, _ = self.make_ui()
        with mock.patch("turbo_core.config.save_setting", return_value="x/.env") as save:
            chat.handle_command("/model code small:1b --save")
        save.assert_called_once_with("TURBO_CODE_MODEL", "small:1b")

    def test_file_attaches_to_next_question(self):
        chat, _, llm, c = self.make_ui("It adds.")
        path = temp_dir(self) / "my file.py"
        path.write_text("a = 1\nb = a + 1\nprint(b)\n", encoding="utf-8")
        chat.handle_command(f"/file {path} 2-3")
        self.assertIn("lines 2-3", output(c))
        chat.ask("what does this do?")
        sent = llm.calls[0]["messages"][-1]["content"]
        self.assertIn("b = a + 1", sent)
        self.assertNotIn("a = 1\n", sent)
        self.assertIsNone(chat.pending_context)

    def test_clear_resets_history(self):
        chat, _, llm, _ = self.make_ui("a", "b")
        chat.ask("q1")
        chat.handle_command("/clear")
        chat.ask("q2")
        self.assertEqual(len(llm.calls[1]["messages"]), 2)

    def test_unknown_command_and_errors_dont_crash(self):
        chat, _, _, c = self.make_ui()
        chat.handle_command("/nope")
        chat.handle_command("/model a b c")
        chat.handle_command("/file")
        self.assertIn("Unknown command", output(c))
        self.assertIn("Usage", output(c))

    def test_exit(self):
        chat, *_ = self.make_ui()
        self.assertEqual(chat.handle_command("/exit"), "exit")

    def test_ask_error_is_shown(self):
        chat, _, _, c = self.make_ui(LLMConnectionError("Can't reach Ollama"))
        chat.ask("hi")
        self.assertIn("Can't reach Ollama", output(c))

    def test_truncated_answer_note(self):
        chat, _, llm, c = self.make_ui("partial answer")
        llm.next_done_reason = "length"
        chat.ask("long question")
        self.assertIn("cut off", output(c))

    def test_models_and_status_render(self):
        chat, _, llm, c = self.make_ui()
        llm.loaded_models = lambda: [{"name": "code-model", "size": 2 * GB, "gpu_percent": 0,
                                      "context": 4096, "expires_at": None}]
        chat.handle_command("/models")
        chat.handle_command("/status")
        self.assertIn("small:1b", output(c))
        self.assertIn("100% on the CPU", output(c))

    def test_banner_warns_about_missing_model(self):
        chat, _, llm, c = self.make_ui()
        llm.model_details = lambda: installed("chat-model")
        chat.print_banner()
        self.assertIn("code model code-model isn't available", output(c))
        self.assertIn("ollama pull code-model", output(c))

    def test_cloud_banner_shows_provider_not_ram(self):
        from dataclasses import replace

        chat, assistant, _, c = self.make_ui()
        assistant.settings = replace(assistant.settings, provider="groq")
        chat.print_banner()
        self.assertIn("Groq (cloud, free tier) connected", output(c))
        self.assertNotIn("RAM", output(c))

    def test_cloud_model_switch_never_offers_download(self):
        from dataclasses import replace

        chat, assistant, _, c = self.make_ui()
        assistant.settings = replace(assistant.settings, provider="groq")
        chat.confirm = lambda *a, **k: self.fail("should not ask to download")
        chat.handle_command("/model not-offered")
        self.assertIn("isn't one of the free models", output(c))

    def test_slash_completer(self):
        from prompt_toolkit.document import Document

        names = [c.text for c in ui.SlashCompleter().get_completions(Document("/mo"), None)]
        self.assertEqual(names, ["/model", "/models"])
        self.assertEqual(list(ui.SlashCompleter().get_completions(Document("hello"), None)), [])


class SpeedTipTests(unittest.TestCase):
    def snap(self, **kw):
        base = dict(ollama_ok=True, installed=installed("llama3.2:3b", "qwen2.5-coder:7b"),
                    chat_model="llama3.2:3b", code_model="qwen2.5-coder:3b",
                    context_tokens=4096, ram_total_gb=16, ram_free_gb=8)
        base.update(kw)
        return system.Snapshot(**base)

    def test_cpu_only_and_big_model(self):
        tips = system.speed_tips(self.snap(
            code_model="qwen2.5-coder:7b",
            installed=[{"name": "qwen2.5-coder:7b", "size": 5 * GB, "parameters": "7.6B"}],
            loaded=[{"name": "qwen2.5-coder:7b", "size": 5 * GB, "gpu_percent": 0, "context": 4096}],
        ))
        self.assertTrue(any("100% on the CPU" in t for t in tips))
        self.assertTrue(any("/model code qwen2.5-coder:3b" in t for t in tips))

    def test_gpu_spill(self):
        tips = system.speed_tips(self.snap(
            loaded=[{"name": "big:14b", "size": 9 * GB, "gpu_percent": 60, "context": 4096}]))
        self.assertTrue(any("spills into CPU RAM" in t for t in tips))

    def test_low_ram(self):
        self.assertTrue(any("RAM" in t for t in system.speed_tips(self.snap(ram_free_gb=1.2))))

    def test_ollama_down(self):
        self.assertIn("isn't running", system.speed_tips(self.snap(ollama_ok=False))[0])

    def test_healthy_machine_has_no_tips(self):
        self.assertEqual(system.speed_tips(self.snap(
            loaded=[{"name": "llama3.2:3b", "size": 2 * GB, "gpu_percent": 100, "context": 4096}])), [])

    def test_model_billions(self):
        self.assertEqual(system.model_billions("qwen2.5-coder:7b"), 7)
        self.assertEqual(system.model_billions("x", [{"name": "x", "parameters": "494M"}]), 0.494)
        self.assertIsNone(system.model_billions("mistral"))

    def test_switch_model_validation(self):
        assistant, _ = make_assistant(self)
        with self.assertRaises(ValueError):
            system.switch_model(assistant, "everything", "x")
        with self.assertRaises(ValueError):
            system.switch_model(assistant, "chat", "two words")
        self.assertIn("chat model", system.switch_model(assistant, "chat", "m:1"))


if __name__ == "__main__":
    unittest.main()
