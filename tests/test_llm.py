import unittest
from unittest import mock

import requests

from turbo_core.llm import (
    LLMBadResponse,
    LLMConnectionError,
    LLMModelNotFound,
    LLMTimeout,
    OllamaClient,
)


def fake_response(status=200, json_data=None, text=""):
    response = mock.Mock()
    response.status_code = status
    response.text = text
    if json_data is None:
        response.json.side_effect = ValueError("no json")
    else:
        response.json.return_value = json_data
    return response


class OllamaClientTests(unittest.TestCase):
    def make_client(self, **kwargs):
        session = mock.Mock()
        client = OllamaClient(session=session, **kwargs)
        return client, session

    def test_returns_stripped_content_and_sends_context_size(self):
        client, session = self.make_client(context_tokens=4096)
        session.post.return_value = fake_response(json_data={"message": {"content": "  hi \n"}})

        answer = client.chat([{"role": "user", "content": "hello"}], model="m")

        self.assertEqual(answer, "hi")
        payload = session.post.call_args.kwargs["json"]
        self.assertEqual(payload["model"], "m")
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["options"]["num_ctx"], 4096)

    @mock.patch("turbo_core.llm.time.sleep")
    def test_retries_once_then_reports_connection_error(self, _sleep):
        client, session = self.make_client(retries=1)
        session.post.side_effect = requests.exceptions.ConnectionError()

        with self.assertRaisesRegex(LLMConnectionError, "Is Ollama running"):
            client.chat([], model="m")
        self.assertEqual(session.post.call_count, 2)

    @mock.patch("turbo_core.llm.time.sleep")
    def test_retry_can_succeed(self, _sleep):
        client, session = self.make_client(retries=1)
        session.post.side_effect = [
            requests.exceptions.ConnectionError(),
            fake_response(json_data={"message": {"content": "ok"}}),
        ]
        self.assertEqual(client.chat([], model="m"), "ok")

    def test_timeout(self):
        client, session = self.make_client()
        session.post.side_effect = requests.exceptions.ReadTimeout()
        with self.assertRaisesRegex(LLMTimeout, "TURBO_TIMEOUT"):
            client.chat([], model="m")

    def test_missing_model_tells_you_how_to_pull_it(self):
        client, session = self.make_client()
        session.post.return_value = fake_response(404, {"error": "model not found"})
        with self.assertRaisesRegex(LLMModelNotFound, "ollama pull qwen"):
            client.chat([], model="qwen")

    def test_server_error(self):
        client, session = self.make_client()
        session.post.return_value = fake_response(500, {"error": "boom"})
        with self.assertRaisesRegex(LLMBadResponse, "500: boom"):
            client.chat([], model="m")

    def test_unreadable_body(self):
        client, session = self.make_client()
        session.post.return_value = fake_response(200, None, text="<html>")
        with self.assertRaises(LLMBadResponse):
            client.chat([], model="m")


class OllamaStreamTests(unittest.TestCase):
    def stream_client(self, lines, status=200):
        session = mock.Mock()
        response = fake_response(status, {"error": "x"} if status >= 400 else None)
        response.iter_lines.return_value = iter(lines)
        session.post.return_value = response
        return OllamaClient(session=session), session, response

    def test_yields_pieces_until_done(self):
        lines = [
            b'{"message": {"content": "Hel"}, "done": false}',
            b"",
            b'{"message": {"content": "lo"}, "done": false}',
            b'{"message": {"content": ""}, "done": true}',
            b'{"message": {"content": "ignored"}}',
        ]
        client, session, response = self.stream_client(lines)
        self.assertEqual(list(client.chat_stream([], model="m")), ["Hel", "lo"])
        self.assertTrue(session.post.call_args.kwargs["json"]["stream"])
        self.assertTrue(session.post.call_args.kwargs["stream"])
        response.close.assert_called()

    def test_error_mid_stream(self):
        client, _, _ = self.stream_client([b'{"message": {"content": "a"}}', b'{"error": "out of memory"}'])
        with self.assertRaisesRegex(LLMBadResponse, "out of memory"):
            list(client.chat_stream([], model="m"))

    def test_bad_line(self):
        client, _, _ = self.stream_client([b"not json"])
        with self.assertRaises(LLMBadResponse):
            list(client.chat_stream([], model="m"))

    def test_stall_is_a_timeout(self):
        client, _, response = self.stream_client([])
        response.iter_lines.side_effect = requests.exceptions.ReadTimeout()
        with self.assertRaises(LLMTimeout):
            list(client.chat_stream([], model="m"))

    def test_missing_model_before_streaming(self):
        client, _, _ = self.stream_client([], status=404)
        with self.assertRaises(LLMModelNotFound):
            list(client.chat_stream([], model="m"))


if __name__ == "__main__":
    unittest.main()
