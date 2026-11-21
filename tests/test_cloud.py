import json
import os
import types
import unittest
from unittest import mock

from tests.helpers import temp_dir
from turbo_core import cloud
from turbo_core.assistant import make_client
from turbo_core.config import Settings
from turbo_core.llm import (LLMAuthError, LLMBadResponse, LLMError, LLMModelNotFound,
                            LLMRateLimited, LLMInputTooLarge)


def response(status=200, body=None, lines=None, headers=None):
    r = mock.Mock()
    r.status_code = status
    r.headers = headers or {}
    r.text = json.dumps(body) if body is not None else ""
    r.json.return_value = body
    if body is None:
        r.json.side_effect = ValueError()
    r.iter_lines.return_value = iter(lines or [])
    return r


def client(**kw):
    session = mock.Mock()
    c = cloud.OpenAICompatClient("groq", "gsk_secret", session=session, **kw)
    return c, session


class ClientTests(unittest.TestCase):
    def test_needs_key_and_hides_it(self):
        with self.assertRaisesRegex(LLMAuthError, "turbo setup"):
            cloud.OpenAICompatClient("groq", "")
        c, _ = client()
        self.assertNotIn("gsk_secret", repr(c))

    def test_chat_sends_openai_request(self):
        c, session = client()
        session.post.return_value = response(body={"choices": [{"message": {"content": " hi "},
                                                                "finish_reason": "stop"}]})
        self.assertEqual(c.chat([{"role": "user", "content": "q"}], model="m", max_tokens=50), "hi")
        args = session.post.call_args
        self.assertEqual(args.args[0], "https://api.groq.com/openai/v1/chat/completions")
        self.assertEqual(args.kwargs["headers"]["Authorization"], "Bearer gsk_secret")
        self.assertEqual(args.kwargs["json"]["max_tokens"], 50)
        self.assertEqual(c.last_done_reason, "stop")

    def test_stream_parses_server_sent_events(self):
        c, session = client()
        session.post.return_value = response(lines=[
            b": keep-alive",
            b'data: {"choices":[{"delta":{"role":"assistant"}}]}',
            b'data: {"choices":[{"delta":{"content":"Hel"}}]}',
            b"",
            b'data: {"choices":[{"delta":{"content":"lo"},"finish_reason":"length"}]}',
            b"data: [DONE]",
        ])
        self.assertEqual(list(c.chat_stream([], model="m")), ["Hel", "lo"])
        self.assertEqual(c.last_done_reason, "length")
        self.assertTrue(session.post.call_args.kwargs["json"]["stream"])

    def test_tool_calls_round_trip(self):
        c, session = client()
        session.post.return_value = response(body={"choices": [{"message": {
            "content": None,
            "tool_calls": [{"id": "x1", "type": "function",
                            "function": {"name": "read_file", "arguments": '{"path": "a.py"}'}}],
        }}]})
        msg = c.chat_tools([{"role": "user", "content": "q"}], model="m", tools=[{"type": "function"}])
        self.assertEqual(msg["tool_calls"], [{"function": {"name": "read_file", "arguments": {"path": "a.py"}}}])

    def test_tool_messages_get_matching_ids(self):
        converted = cloud._to_openai_messages([
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "a", "arguments": {}}}, {"function": {"name": "b", "arguments": {"x": 1}}}]},
            {"role": "tool", "tool_name": "a", "content": "ra"},
            {"role": "tool", "tool_name": "b", "content": "rb"},
        ])
        ids = [call["id"] for call in converted[1]["tool_calls"]]
        self.assertEqual([converted[2]["tool_call_id"], converted[3]["tool_call_id"]], ids)
        self.assertEqual(converted[1]["tool_calls"][1]["function"]["arguments"], '{"x": 1}')
        self.assertNotIn("tool_name", converted[2])

    def test_error_mapping(self):
        cases = [
            (401, LLMAuthError, "turbo setup"),
            (402, LLMBadResponse, "free"),
            (404, LLMModelNotFound, "/models"),
            (413, LLMInputTooLarge, "too much"),
            (500, LLMBadResponse, "HTTP 500"),
        ]
        for status, error, text in cases:
            c, session = client()
            session.post.return_value = response(status, {"error": {"message": "nope"}})
            with self.assertRaisesRegex(error, text):
                c.chat([], model="m")

    def test_rate_limit_says_when_to_retry(self):
        c, session = client()
        session.post.return_value = response(429, {"error": {"message": "Rate limit reached"}},
                                             headers={"retry-after": "120"})
        with self.assertRaisesRegex(LLMRateLimited, "2 minutes"):
            c.chat([], model="m")

    def test_models_list_drops_non_chat(self):
        c, session = client()
        session.get.return_value = response(body={"data": [
            {"id": "whisper-large-v3"}, {"id": "openai/gpt-oss-20b"}, {"id": "llama-guard-4"},
            {"id": "qwen/qwen3-32b"}, {"id": "playai-tts"}, {"id": "canopylabs/orpheus-v1-english"}]})
        self.assertEqual(c.list_models(), ["openai/gpt-oss-20b", "qwen/qwen3-32b"])

    def test_no_local_operations(self):
        c, _ = client()
        self.assertEqual(c.loaded_models(), [])
        c.warm("m")
        with self.assertRaisesRegex(LLMError, "only for local Ollama"):
            c.pull("m")
        with self.assertRaises(LLMError):
            c.unload("m")


class CostSafetyTests(unittest.TestCase):
    def test_openrouter_only_free_models(self):
        ids = ["meta/llama-3.3-70b", "qwen/qwen3-coder:free", "openai/gpt-4o", "google/gemma:free"]
        self.assertEqual(cloud.usable_models("openrouter", ids), ["google/gemma:free", "qwen/qwen3-coder:free"])

    def test_pick_models_prefers_coder_for_code_and_small_for_chat(self):
        ids = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3-32b", "whisper-large-v3"]
        chat, code = cloud.pick_models("groq", ids)
        self.assertEqual(chat, "openai/gpt-oss-20b")
        self.assertEqual(code, "qwen/qwen3-32b")

    def test_pick_models_falls_back_to_anything(self):
        self.assertEqual(cloud.pick_models("groq", ["some-model"]), ("some-model", "some-model"))
        self.assertEqual(cloud.pick_models("groq", []), (None, None))


class SettingsTests(unittest.TestCase):
    def test_provider_key_and_fallback_env(self):
        with mock.patch.dict(os.environ, {"USERPROFILE": os.environ.get("USERPROFILE", ""),
                                          "HOME": os.environ.get("HOME", ""),
                                          "TURBO_PROVIDER": "Groq", "GROQ_API_KEY": "k1"}, clear=True):
            settings = Settings.from_env(dotenv=False)
        self.assertEqual(settings.provider, "groq")
        self.assertTrue(settings.is_cloud)
        self.assertEqual(settings.api_key, "k1")
        self.assertNotIn("k1", repr(settings))

    def test_make_client_picks_backend(self):
        cloud_client = make_client(Settings(provider="groq", api_key="k", db_path=temp_dir(self) / "x.db"))
        self.assertIsInstance(cloud_client, cloud.OpenAICompatClient)
        local = make_client(Settings(db_path=temp_dir(self) / "x.db"))
        self.assertEqual(type(local).__name__, "OllamaClient")


class SetupTests(unittest.TestCase):
    def run_setup(self, answers, secrets, models, test_reply="ok", provider=None):
        home = temp_dir(self)
        out = []
        fake = mock.Mock()
        fake.list_models.return_value = models
        fake.chat.return_value = test_reply
        env = {"TURBO_HOME": str(home), "USERPROFILE": os.environ.get("USERPROFILE", ""),
               "HOME": os.environ.get("HOME", "")}
        from turbo_core import setup
        with mock.patch.dict(os.environ, env, clear=True), \
                mock.patch.object(setup, "OpenAICompatClient", return_value=fake):
            code = setup.run_setup(types.SimpleNamespace(provider=provider),
                                   input_func=lambda _p: answers.pop(0),
                                   secret_func=lambda _p: secrets.pop(0), out=out.append)
        env_file = home / ".env"
        return code, "\n".join(out), env_file.read_text() if env_file.exists() else ""

    def test_groq_setup_saves_provider_key_and_models(self):
        code, out, saved = self.run_setup(["1"], ["gsk_abc"], ["openai/gpt-oss-20b", "qwen/qwen3-32b"])
        self.assertEqual(code, 0, out)
        self.assertIn("TURBO_PROVIDER=groq", saved)
        self.assertIn("TURBO_API_KEY=gsk_abc", saved)
        self.assertIn("TURBO_CHAT_MODEL=openai/gpt-oss-20b", saved)
        self.assertIn("TURBO_CODE_MODEL=qwen/qwen3-32b", saved)
        self.assertIn("Do NOT add a credit card", out)
        self.assertNotIn("gsk_abc", out)  # key never echoed

    def test_bad_key_retries_then_gives_up_without_saving(self):
        from turbo_core import setup
        home = temp_dir(self)
        with mock.patch.dict(os.environ, {"TURBO_HOME": str(home)}), \
                mock.patch.object(setup, "OpenAICompatClient", side_effect=LLMAuthError("bad")):
            code = setup.run_setup(types.SimpleNamespace(provider="groq"), input_func=lambda _p: "",
                                   secret_func=lambda _p: "wrong", out=lambda *_: None)
        self.assertEqual(code, 1)
        self.assertFalse((home / ".env").exists())

    def test_choose_ollama_resets_to_local(self):
        code, out, saved = self.run_setup(["4"], [], [])
        self.assertEqual(code, 0)
        self.assertIn("TURBO_PROVIDER=ollama", saved)
        self.assertIn("TURBO_CODE_MODEL=qwen2.5-coder:3b", saved)


if __name__ == "__main__":
    unittest.main()
