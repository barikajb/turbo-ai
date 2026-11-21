import json
import unittest
from pathlib import Path
from unittest import mock

from tests.helpers import make_assistant, temp_dir
from turbo_core import manager as mgr
from turbo_core.system import GB


class FakeUI:
    def __init__(self, answers=None):
        self.answers = list(answers or [])
        self.asked, self.notes, self.final, self.diffs = [], [], [], []
        self.pulled = []

    def confirm(self, question, default=False):
        self.asked.append(question)
        return self.answers.pop(0) if self.answers else default

    def note(self, text, style=None):
        self.notes.append(text)

    def wait(self, fn, label=""):
        return fn()

    def markdown(self, text):
        self.final.append(text)

    def warm(self, model):
        pass

    def pull(self, name):
        self.pulled.append(name)
        return True

    def show_diff(self, diff):
        self.diffs.append(diff)

    def show_output(self, text):
        pass


def model(name, params, size=2):
    return {"name": name, "size": size * GB, "parameters": params, "quantization": "Q4"}


class Base(unittest.TestCase):
    def setUp(self):
        self.assistant, self.llm = make_assistant(self)
        self.llm.model_details = lambda: [model("chat-model", "3.2B"), model("code-model", "3.1B"),
                                          model("qwen2.5-coder:3b", "3.1B"), model("big:7b", "7.6B", 5)]
        self.llm.list_models = lambda: [m["name"] for m in self.llm.model_details()]
        self.llm.loaded_models = lambda: []
        self.llm.supports_tools = lambda name: True
        self.tool_replies = []
        self.llm.chat_tools = self.fake_chat_tools
        self.tool_calls_sent = []
        self.manager = mgr.Manager(self.assistant)

    def fake_chat_tools(self, messages, model, tools, **kw):
        self.tool_calls_sent.append({"messages": list(messages), "model": model, "tools": tools})
        return self.tool_replies.pop(0)


def call(name, **args):
    return {"content": "", "tool_calls": [{"function": {"name": name, "arguments": args}}]}


class ClassifyTests(unittest.TestCase):
    def test_routes(self):
        cases = {
            "it's too slow, switch to a faster model": "speed",
            "speed it up": "speed",
            "check my setup": "diagnose",
            "fix a dependency issue": "agent",
            "switch code to qwen2.5-coder:1.5b": "agent",
            "why does turbo.py crash": "agent",
            "tell me a joke": "agent",
            "which model is best for coding? check what is available first": "agent",
            "why is it so slow?": "agent",
            "what's wrong with my setup": "diagnose",
            "is everything ok?": "diagnose",
            "make it faster": "speed",
        }
        for text, expected in cases.items():
            self.assertEqual(mgr.classify(text), expected, text)


class AgentLoopTests(Base):
    def test_tool_then_answer(self):
        self.tool_replies = [call("get_status"), {"content": "All fine."}]
        ui = FakeUI()
        answer = self.manager.run("tell me about my models", ui)
        self.assertEqual(answer, "All fine.")
        tool_msg = self.tool_calls_sent[1]["messages"][-1]
        self.assertEqual(tool_msg["role"], "tool")
        self.assertIn("installed_models", tool_msg["content"])
        self.assertEqual(self.tool_calls_sent[0]["model"], "chat-model")

    def test_repeated_call_forces_summary_instead_of_looping(self):
        self.tool_replies = [call("get_status"), call("get_status"), {"content": "Summary."}]
        ui = FakeUI()
        self.assertEqual(self.manager.run("tell me about my models", ui), "Summary.")
        self.assertEqual(self.tool_calls_sent[-1]["tools"], [])  # summary call has no tools

    def test_json_in_text_is_understood(self):
        self.tool_replies = [{"content": '```json\n{"name": "get_status", "arguments": {}}\n```'},
                             {"content": "ok"}]
        self.manager.run("tell me about my models", FakeUI())
        self.assertEqual(self.tool_calls_sent[1]["messages"][-1]["tool_name"], "get_status")

    def test_unknown_tool_and_bad_args_are_reported_to_model(self):
        self.tool_replies = [call("format_disk"), call("read_file", nonsense=1), {"content": "done"}]
        self.manager.run("tell me about my models", FakeUI())
        results = [m["content"] for m in self.tool_calls_sent[2]["messages"] if m["role"] == "tool"]
        self.assertIn("Unknown tool", results[0])
        self.assertIn("Bad arguments", results[1])

    def test_falls_back_to_code_model_without_tool_support(self):
        self.llm.supports_tools = lambda name: name != "chat-model"
        self.assertEqual(self.manager.model, "code-model")


class ChangeToolTests(Base):
    def run_tool(self, tool_name, answers=None, task="do it", /, **args):
        self.manager.ui = FakeUI(answers)
        self.manager.task = task
        return self.manager.call_tool(tool_name, args), self.manager.ui

    def test_switch_model_asks_and_does_not_save_unless_asked(self):
        with mock.patch("turbo_core.config.save_setting") as save:
            result, ui = self.run_tool("switch_model", [True], role="code", name="big:7b", save=True)
        self.assertTrue(result["ok"])
        self.assertEqual(self.assistant.settings.code_model, "big:7b")
        save.assert_not_called()
        self.assertNotIn("save", ui.asked[0])

    def test_switch_model_saves_when_user_said_so(self):
        with mock.patch("turbo_core.config.save_setting", return_value=Path("x")) as save:
            self.run_tool("switch_model", [True], "use big:7b and keep it as default",
                          role="code", name="big:7b", save=True)
        save.assert_called_once_with("TURBO_CODE_MODEL", "big:7b")

    def test_switch_declined_changes_nothing(self):
        result, _ = self.run_tool("switch_model", [False], role="code", name="big:7b")
        self.assertTrue(result["declined"])
        self.assertEqual(self.assistant.settings.code_model, "code-model")

    def test_switch_to_current_model_is_a_no_op(self):
        result, ui = self.run_tool("switch_model", role="code", name="code-model")
        self.assertIn("nothing to change", result["note"])
        self.assertEqual(ui.asked, [])

    def test_switch_to_missing_model(self):
        result, _ = self.run_tool("switch_model", role="code", name="nope:1b")
        self.assertIn("pull_model", result["error"])

    def test_change_setting_validates_saves_and_applies(self):
        home = temp_dir(self)
        with mock.patch.dict("os.environ", {"TURBO_HOME": str(home)}):
            bad, _ = self.run_tool("change_setting", key="PATH", value="x")
            self.assertIn("can't be changed", bad["error"])
            bad, _ = self.run_tool("change_setting", key="TURBO_MAX_TOKENS", value="lots")
            self.assertIn("whole number", bad["error"])
            result, _ = self.run_tool("change_setting", [True], key="TURBO_MAX_TOKENS", value="500")
        self.assertEqual(result["applies"], "now")
        self.assertEqual(self.assistant.settings.max_tokens, 500)
        self.assertIn("TURBO_MAX_TOKENS=500", (home / ".env").read_text())

    def test_run_command_needs_yes(self):
        with mock.patch("subprocess.run") as run:
            result, _ = self.run_tool("run_command", [False], command="pip install x", reason="r")
        self.assertTrue(result["declined"])
        run.assert_not_called()

    def test_run_command_runs_after_yes(self):
        done = mock.Mock(returncode=0, stdout="installed", stderr="")
        with mock.patch("subprocess.run", return_value=done) as run:
            result, ui = self.run_tool("run_command", [True], command="echo hi", reason="test")
        self.assertEqual(result["exit_code"], 0)
        self.assertIn("echo hi", ui.asked[0])
        self.assertEqual(run.call_args.args[0], "echo hi")


class FileToolTests(Base):
    def setUp(self):
        super().setUp()
        self.root = temp_dir(self)
        self.file = self.root / "notes.txt"
        self.file.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
        patcher = mock.patch.object(mgr, "package_root", return_value=self.root.resolve())
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_tool(self, tool_name, answers=None, /, **args):
        self.manager.ui = FakeUI(answers)
        self.manager.task = "x"
        return self.manager.call_tool(tool_name, args), self.manager.ui

    def test_read_file_with_line_numbers(self):
        result, _ = self.run_tool("read_file", path="notes.txt", start_line=2, end_line=3)
        self.assertIn("    2  beta", result["text"])
        self.assertNotIn("alpha", result["text"])

    def test_paths_outside_turbo_are_refused(self):
        outside = temp_dir(self) / "secret.txt"
        outside.write_text("x")
        for path in (str(outside), "../../etc/passwd"):
            result, _ = self.run_tool("read_file", path=path)
            self.assertIn("outside TURBO", result["error"])

    def test_secret_files_are_refused(self):
        (self.root / "credentials.json").write_text("{}")
        result, _ = self.run_tool("read_file", path="credentials.json")
        self.assertIn("off limits", result["error"])

    def test_edit_file_shows_diff_and_needs_yes(self):
        result, ui = self.run_tool("edit_file", [False], path="notes.txt", old_text="beta", new_text="BETA")
        self.assertTrue(result["declined"])
        self.assertIn("-beta", ui.diffs[0])
        self.assertIn("beta", self.file.read_text())
        result, _ = self.run_tool("edit_file", [True], path="notes.txt", old_text="beta", new_text="BETA")
        self.assertTrue(result["ok"])
        self.assertEqual(self.file.read_text(), "alpha\nBETA\ngamma\n")

    def test_edit_requires_unique_match(self):
        result, _ = self.run_tool("edit_file", path="notes.txt", old_text="a", new_text="b")
        self.assertIn("exactly once", result["error"])

    def test_list_files(self):
        result, _ = self.run_tool("list_files")
        self.assertTrue(any("notes.txt" in e for e in result["entries"]))


class PlaybookTests(Base):
    def test_speed_switches_only_big_role_without_saving(self):
        self.assistant.set_models(code="big:7b")
        ui = FakeUI([True])
        with mock.patch("turbo_core.config.save_setting") as save:
            self.manager.run("it's too slow", ui)
        self.assertEqual(self.assistant.settings.code_model, "qwen2.5-coder:3b")
        self.assertEqual(self.assistant.settings.chat_model, "chat-model")
        save.assert_not_called()
        self.assertIn("big:7b → qwen2.5-coder:3b", ui.final[0])
        self.assertEqual(self.tool_calls_sent, [])  # no model calls needed

    def test_speed_when_already_fast(self):
        ui = FakeUI()
        self.manager.run("make it faster", ui)
        self.assertIn("already on the fast settings", ui.final[0])
        self.assertEqual(ui.asked, [])

    def test_speed_offers_download_when_no_fast_model_installed(self):
        self.llm.model_details = lambda: [model("chat-model", "3.2B"), model("big:7b", "7.6B", 5)]
        self.assistant.set_models(code="big:7b")
        ui = FakeUI([True, True])
        self.manager.run("too slow", ui)
        self.assertEqual(ui.pulled, ["qwen2.5-coder:3b"])

    def test_speed_unloads_unused_models_when_ram_is_low(self):
        self.llm.loaded_models = lambda: [{"name": "big:7b", "size": 5 * GB, "gpu_percent": 0,
                                           "context": 4096, "expires_at": None}]
        unloaded = []
        self.llm.unload = unloaded.append
        with mock.patch.object(mgr.system, "psutil") as ps:
            ps.virtual_memory.return_value = mock.Mock(total=16 * GB, available=1 * GB)
            ps.cpu_percent.return_value = 10.0
            self.manager.run("so slow", FakeUI([True]))
        self.assertEqual(unloaded, ["big:7b"])

    def test_diagnose_reports_missing_model_and_dependency(self):
        self.assistant.set_models(code="missing:3b")
        ui = FakeUI([False, False])  # decline pip install and download
        with mock.patch("importlib.metadata.version", side_effect=lambda p:
                        (_ for _ in ()).throw(mgr.importlib.metadata.PackageNotFoundError()) if p == "psutil"
                        else "1.0"):
            self.manager.run("check my setup", ui)
        report = ui.final[0]
        self.assertIn("psutil", report)
        self.assertIn("ollama pull missing:3b", report)
        self.assertIn("pip install psutil", ui.asked[0])


class ExtractTests(unittest.TestCase):
    def test_string_arguments_are_parsed(self):
        msg = {"tool_calls": [{"function": {"name": "read_file", "arguments": json.dumps({"path": "a"})}}]}
        self.assertEqual(mgr.extract_tool_calls(msg), [("read_file", {"path": "a"})])

    def test_plain_text_is_not_a_call(self):
        self.assertEqual(mgr.extract_tool_calls({"content": "Use {braces} carefully."}), [])


if __name__ == "__main__":
    unittest.main()
