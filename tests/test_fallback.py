import io
import json
import types
import unittest
from unittest import mock

from tests.helpers import make_assistant
from turbo_core import fallback
from turbo_core.fallback import FallbackClient, LimitReached, RequestCancelled, run_with_choices
from turbo_core.llm import LLMInputTooLarge, LLMRateLimited

CODE, CHAT = "qwen/qwen3.8-27b", "openai/gpt-oss-20b"
MODELS = ["allam-2-7b", "openai/gpt-oss-120b", CHAT, CODE]


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def limited(model, wait=30.0):
    error = LLMRateLimited(f"{model} reached the free limit")
    error.model, error.retry_after = model, wait
    return error


class FakeCloud:
    provider = "groq"

    def __init__(self):
        self.calls = []
        self.fail = {}      # model -> exception to raise once
        self.limits = {}
        self.last_done_reason = "stop"

    def list_models(self):
        return MODELS

    def _maybe_fail(self, model):
        self.calls.append(model)
        if model in self.fail:
            raise self.fail.pop(model)

    def chat(self, messages, model, **kw):
        self._maybe_fail(model)
        return f"answer from {model}"

    def chat_tools(self, messages, model, tools, **kw):
        self._maybe_fail(model)
        return {"content": f"tools from {model}", "tool_calls": []}

    def chat_stream(self, messages, model, **kw):
        self._maybe_fail(model)
        yield from ["from ", model]


class FakeLocal:
    def __init__(self, models=("qwen2.5-coder:3b", "llama3.2:3b")):
        self.models = list(models)
        self.calls = []
        self.last_done_reason = "stop"

    def list_models(self):
        return self.models

    def chat(self, messages, model, **kw):
        self.calls.append(model)
        return f"local answer from {model}"


def msgs(chars=100):
    return [{"role": "user", "content": "x" * chars}]


class Base(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.cloud = FakeCloud()
        self.local = FakeLocal()
        self.client = FallbackClient(self.cloud, local_factory=lambda: self.local, clock=self.clock)
        self.settings = types.SimpleNamespace(chat_model=CHAT, code_model=CODE)
        self.client.bind(lambda: self.settings)


class LimitTests(Base):
    def test_normal_requests_pass_straight_through(self):
        self.assertEqual(self.client.chat(msgs(), CODE), f"answer from {CODE}")
        self.assertIsNone(self.client.last_route_note)
        self.assertIn(("cloud", CODE), self.client.speeds)

    def test_limit_offers_next_model_local_wait_and_cancel_with_estimates(self):
        self.cloud.fail[CODE] = limited(CODE, wait=35)
        with self.assertRaises(LimitReached) as caught:
            self.client.chat(msgs(), CODE)
        decision = caught.exception.decision
        self.assertIn("resets in about 35s", decision.message)
        kinds = [(o.kind, o.model, o.backend) for o in decision.options]
        self.assertEqual(kinds, [
            ("switch", "openai/gpt-oss-120b", "cloud"),      # best other free model for code
            ("switch", "qwen2.5-coder:3b", "local"),
            ("wait", CODE, "cloud"),
            ("cancel", None, "cloud"),
        ])
        wait = decision.options[2]
        self.assertEqual(wait.wait_seconds, 35)
        self.assertAlmostEqual(wait.estimate_seconds, 35 + fallback.DEFAULT_SECONDS["cloud"])
        self.assertEqual(decision.options[1].estimate_seconds, fallback.DEFAULT_SECONDS["local"])

    def test_switch_is_remembered_until_reset_then_returns(self):
        self.cloud.fail[CODE] = limited(CODE, wait=30)
        with self.assertRaises(LimitReached) as caught:
            self.client.chat(msgs(), CODE)
        self.client.choose(caught.exception.decision, 0)

        self.assertEqual(self.client.chat(msgs(), CODE), "answer from openai/gpt-oss-120b")
        self.assertIn("while qwen/qwen3.8-27b resets", self.client.last_route_note)

        self.clock.now += 31
        self.assertEqual(self.client.chat(msgs(), CODE), f"answer from {CODE}")
        self.assertEqual(self.client.last_route_note, f"back to {CODE}")

    def test_known_limit_asks_without_spending_a_request(self):
        self.cloud.fail[CODE] = limited(CODE, wait=60)
        with self.assertRaises(LimitReached):
            self.client.chat(msgs(), CODE)
        calls = len(self.cloud.calls)
        with self.assertRaises(LimitReached):
            self.client.chat(msgs(), CODE)
        self.assertEqual(len(self.cloud.calls), calls)

    def test_local_fallback(self):
        self.cloud.fail[CODE] = limited(CODE)
        with self.assertRaises(LimitReached) as caught:
            self.client.chat(msgs(), CODE)
        local_option = next(o for o in caught.exception.decision.options if o.backend == "local")
        self.client.choose(caught.exception.decision, local_option)
        self.assertEqual(self.client.chat(msgs(), CODE), "local answer from qwen2.5-coder:3b")

    def test_no_local_option_without_ollama(self):
        client = FallbackClient(self.cloud, local_factory=lambda: (_ for _ in ()).throw(OSError()),
                                clock=self.clock)
        client.bind(lambda: self.settings)
        self.cloud.fail[CODE] = limited(CODE)
        with self.assertRaises(LimitReached) as caught:
            client.chat(msgs(), CODE)
        self.assertFalse(any(o.backend == "local" for o in caught.exception.decision.options))

    def test_chat_role_uses_chat_ranking(self):
        self.cloud.fail[CHAT] = limited(CHAT)
        with self.assertRaises(LimitReached) as caught:
            self.client.chat(msgs(), CHAT)
        first = caught.exception.decision.options[0]
        self.assertEqual(first.model, "openai/gpt-oss-120b")
        local = next(o for o in caught.exception.decision.options if o.backend == "local")
        self.assertEqual(local.model, "llama3.2:3b")

    def test_measured_speed_replaces_default_estimate(self):
        def slow_chat(messages, model, **kw):
            self.clock.now += 1.5
            return "ok"
        self.cloud.chat = slow_chat
        self.client.chat(msgs(), "openai/gpt-oss-120b")
        self.cloud.chat = FakeCloud.chat.__get__(self.cloud)
        self.cloud.fail[CODE] = limited(CODE)
        with self.assertRaises(LimitReached) as caught:
            self.client.chat(msgs(), CODE)
        self.assertAlmostEqual(caught.exception.decision.options[0].estimate_seconds, 1.5)

    def test_streaming_limit_before_first_word(self):
        self.cloud.fail[CODE] = limited(CODE)
        with self.assertRaises(LimitReached):
            list(self.client.chat_stream(msgs(), CODE))
        self.assertEqual("".join(self.client.chat_stream(msgs(), "openai/gpt-oss-120b")),
                         "from openai/gpt-oss-120b")

    def test_tools_calls_fall_back_too(self):
        self.cloud.fail[CHAT] = limited(CHAT)
        with self.assertRaises(LimitReached):
            self.client.chat_tools(msgs(), CHAT, [])

    def test_cancel(self):
        self.cloud.fail[CODE] = limited(CODE)
        with self.assertRaises(LimitReached) as caught:
            self.client.chat(msgs(), CODE)
        with self.assertRaises(RequestCancelled):
            self.client.choose(caught.exception.decision, "cancel")

    def test_choose_by_decision_id_for_the_server(self):
        self.cloud.fail[CODE] = limited(CODE)
        with self.assertRaises(LimitReached) as caught:
            self.client.chat(msgs(), CODE)
        chosen = self.client.choose(caught.exception.decision.id, 0)
        self.assertEqual(chosen.model, "openai/gpt-oss-120b")
        with self.assertRaisesRegex(Exception, "expired"):
            self.client.choose("unknown-id", 0)


class TooLargeTests(Base):
    def setUp(self):
        super().setUp()
        self.cloud.limits = {m: {"limit_tokens": 8000} for m in MODELS}

    def test_warns_before_sending_and_recommends_less_code(self):
        with self.assertRaises(LimitReached) as caught:
            self.client.chat(msgs(30_000), CODE)  # ~10k tokens
        decision = caught.exception.decision
        self.assertEqual(decision.kind, "too_large")
        self.assertIn("8,000 tokens per minute", decision.message)
        self.assertIn("Selecting less code", decision.advice)
        self.assertEqual(self.cloud.calls, [])  # nothing was sent
        # No other Groq model has a bigger limit, so: local Ollama or cancel.
        self.assertEqual([(o.kind, o.backend) for o in decision.options],
                         [("switch", "local"), ("cancel", "cloud")])

    def test_one_shot_local_for_that_request_only(self):
        with self.assertRaises(LimitReached) as caught:
            self.client.chat(msgs(30_000), CODE)
        self.client.choose(caught.exception.decision, 0)
        self.assertIn("local answer", self.client.chat(msgs(30_000), CODE))
        self.assertEqual(self.client.chat(msgs(), CODE), f"answer from {CODE}")

    def test_bigger_model_offered_when_it_fits(self):
        self.cloud.limits["openai/gpt-oss-120b"] = {"limit_tokens": 30_000}
        with self.assertRaises(LimitReached) as caught:
            self.client.chat(msgs(30_000), CODE)
        self.assertEqual(caught.exception.decision.options[0].model, "openai/gpt-oss-120b")

    def test_provider_413_becomes_the_same_question(self):
        error = LLMInputTooLarge("too large")
        error.model, error.limit, error.requested = CODE, 8000, 9200
        self.cloud.limits = {}
        self.cloud.fail[CODE] = error
        with self.assertRaises(LimitReached) as caught:
            self.client.chat(msgs(), CODE)
        self.assertIn("9,200 tokens", caught.exception.decision.message)


class UnknownLimitTests(Base):
    def test_unknown_model_limit_assumes_lowest_seen(self):
        self.cloud.limits = {CHAT: {"limit_tokens": 8000}}  # only the chat model's limit is known
        with self.assertRaises(LimitReached) as caught:
            self.client.chat(msgs(30_000), CODE)
        self.assertEqual(self.cloud.calls, [])  # caught before sending
        self.assertFalse(any(o.backend == "cloud" and o.kind == "switch"
                             for o in caught.exception.decision.options))

    def test_provider_limit_is_remembered(self):
        error = LLMInputTooLarge("too large")
        error.model, error.limit, error.requested = CODE, 7000, 9200
        self.cloud.fail[CODE] = error
        with self.assertRaises(LimitReached):
            self.client.chat(msgs(), CODE)
        self.assertEqual(self.cloud.limits[CODE]["limit_tokens"], 7000)


class RunWithChoicesTests(Base):
    def test_wait_then_retry(self):
        self.cloud.fail[CODE] = limited(CODE, wait=20)
        waited = []

        def wait(seconds, model):
            waited.append((seconds, model))
            self.clock.now += seconds

        ask = lambda d: next(o for o in d.options if o.kind == "wait")
        result = run_with_choices(lambda: self.client.chat(msgs(), CODE), self.client, ask, wait)
        self.assertEqual(result, f"answer from {CODE}")
        self.assertEqual(waited, [(20, CODE)])

    def test_switch_then_retry(self):
        self.cloud.fail[CODE] = limited(CODE)
        result = run_with_choices(lambda: self.client.chat(msgs(), CODE), self.client,
                                  lambda d: d.options[0], lambda s, m: None)
        self.assertEqual(result, "answer from openai/gpt-oss-120b")

    def test_none_means_cancel(self):
        self.cloud.fail[CODE] = limited(CODE)
        with self.assertRaises(RequestCancelled):
            run_with_choices(lambda: self.client.chat(msgs(), CODE), self.client,
                             lambda d: None, lambda s, m: None)


class CloudHeaderTests(unittest.TestCase):
    def make(self, status, body, headers):
        from turbo_core.cloud import OpenAICompatClient

        session = mock.Mock()
        response = mock.Mock(status_code=status, headers=headers, text=json.dumps(body))
        response.json.return_value = body
        session.post.return_value = response
        return OpenAICompatClient("groq", "k", session=session)

    def test_records_limits_from_headers(self):
        client = self.make(200, {"choices": [{"message": {"content": "ok"}}]}, {
            "x-ratelimit-limit-requests": "1000", "x-ratelimit-remaining-requests": "997",
            "x-ratelimit-limit-tokens": "8000", "x-ratelimit-remaining-tokens": "7830",
            "x-ratelimit-reset-requests": "4m19.2s", "x-ratelimit-reset-tokens": "1.275s"})
        client.chat([], model="m")
        info = client.limits["m"]
        self.assertEqual((info["remaining_requests"], info["limit_tokens"]), (997, 8000))
        self.assertAlmostEqual(info["reset_requests"], 259.2)

    def test_429_carries_retry_after(self):
        client = self.make(429, {"error": {"message": "Rate limit reached"}}, {"retry-after": "12"})
        with self.assertRaises(LLMRateLimited) as caught:
            client.chat([], model="m")
        self.assertEqual(caught.exception.retry_after, 12)

    def test_413_tpm_message_is_parsed(self):
        client = self.make(413, {"error": {"message":
            "Request too large for model `m` on tokens per minute (TPM): Limit 8000, Requested 9200"}}, {})
        with self.assertRaises(LLMInputTooLarge) as caught:
            client.chat([], model="m")
        self.assertEqual((caught.exception.limit, caught.exception.requested), (8000, 9200))
        self.assertIn("Selecting less code", str(caught.exception))


class InterfaceTests(Base):
    def test_ui_question_and_choice(self):
        from rich.console import Console
        from turbo_core.ui import ChatUI

        assistant, _ = make_assistant(self)
        assistant.llm = self.client
        out = io.StringIO()
        chat = ChatUI(assistant, "s", console=Console(file=out, width=120, force_terminal=False))
        self.cloud.fail[CODE] = limited(CODE, wait=35)
        with self.assertRaises(LimitReached) as caught:
            self.client.chat(msgs(), CODE)
        with mock.patch.object(chat.console, "input", return_value="3"):
            picked = chat.choose(caught.exception.decision)
        self.assertEqual(picked.kind, "wait")
        text = out.getvalue()
        self.assertIn("reached its free limit", text)
        self.assertIn("1. Continue now with openai/gpt-oss-120b", text)
        self.assertIn("~3s to finish", text)
        self.assertIn("~38s to finish", text)

    def test_cli_non_interactive_picks_first_option(self):
        from turbo_core import cli

        self.cloud.fail[CODE] = limited(CODE)
        with self.assertRaises(LimitReached) as caught:
            self.client.chat(msgs(), CODE)
        err = io.StringIO()
        with mock.patch("sys.stderr", err), mock.patch("sys.stdin", io.StringIO()):
            picked = cli.ask_choice(caught.exception.decision)
        self.assertEqual(picked.model, "openai/gpt-oss-120b")
        self.assertIn("not interactive", err.getvalue())

    def test_server_returns_decision_and_accepts_choice(self):
        import threading
        import urllib.error
        import urllib.request
        from dataclasses import replace
        from turbo_core.server import create_server

        assistant, _ = make_assistant(self)
        assistant.llm = self.client
        self.settings.code_model = CODE
        assistant.settings = replace(assistant.settings, code_model=CODE, port=0)
        server = create_server(assistant, assistant.settings)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        base = f"http://127.0.0.1:{server.server_address[1]}"

        def post(path, body):
            request = urllib.request.Request(base + path, data=json.dumps(body).encode(), method="POST",
                                             headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(request, timeout=10) as response:
                    return response.status, json.loads(response.read())
            except urllib.error.HTTPError as error:
                return error.code, json.loads(error.read())

        self.cloud.fail[CODE] = limited(CODE)
        status, body = post("/explain", {"code": "x = 1"})
        self.assertEqual(status, 429)
        self.assertEqual(body["decision"]["options"][0]["model"], "openai/gpt-oss-120b")
        status, chosen = post("/choose", {"decision_id": body["decision"]["id"], "option": 0})
        self.assertEqual(status, 200)
        status, body = post("/explain", {"code": "x = 1"})
        self.assertEqual((status, body["answer"]), (200, "answer from openai/gpt-oss-120b"))


if __name__ == "__main__":
    unittest.main()
