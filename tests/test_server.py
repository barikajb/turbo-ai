import json
import threading
import unittest
import urllib.error
import urllib.request
from dataclasses import replace

from tests.helpers import make_assistant
from tests.test_assistant import BUGGY, FIX_REPLY
from turbo_core.llm import LLMConnectionError, LLMModelNotFound
from turbo_core.server import create_server


class ServerTests(unittest.TestCase):
    def start(self, *replies, token=""):
        assistant, llm = make_assistant(self, *replies)
        settings = replace(assistant.settings, host="127.0.0.1", port=0, token=token)
        assistant.settings = settings
        server = create_server(assistant, settings)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        self.base = f"http://127.0.0.1:{server.server_address[1]}"
        return llm

    def call(self, method, path, body=None, headers=None, raw=None):
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        all_headers = {"Content-Type": "application/json"}
        all_headers.update(headers or {})
        request = urllib.request.Request(self.base + path, data=data, method=method, headers=all_headers)
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read())

    def test_ask_with_follow_up(self):
        llm = self.start("first", "second")
        self.assertEqual(self.call("POST", "/ask", {"question": "q1", "session_id": "v"}), (200, {"answer": "first"}))
        self.call("POST", "/ask", {"question": "q2", "session_id": "v"})
        self.assertEqual(len(llm.calls[1]["messages"]), 4)

    def test_fix_returns_diff(self):
        self.start(FIX_REPLY)
        status, body = self.call("POST", "/fix", {"code": BUGGY, "filename": "stats.py", "start_line": 5})
        self.assertEqual(status, 200)
        self.assertTrue(body["changed"])
        self.assertIn("@@ -5,2 +5,2 @@", body["diff"])
        self.assertEqual(body["fixed_code"], "def average(xs):\n    return sum(xs) / len(xs)\n")

    def test_explain_and_reset(self):
        self.start("It averages.")
        self.assertEqual(self.call("POST", "/explain", {"code": BUGGY})[1], {"answer": "It averages."})
        self.assertEqual(self.call("POST", "/reset", {})[1], {"cleared": 2})

    def test_health_when_ollama_is_down(self):
        self.start()
        status, body = self.call("GET", "/health")
        self.assertEqual(status, 200)
        self.assertTrue(body["ok"])
        self.assertFalse(body["ollama"])  # FakeLLM has no list_models

    def test_input_errors_are_400(self):
        self.start()
        self.assertEqual(self.call("POST", "/fix", {"code": ""})[0], 400)
        self.assertEqual(self.call("POST", "/fix", {"code": "x", "start_line": "1"})[0], 400)
        self.assertEqual(self.call("POST", "/ask", {"question": 5})[0], 400)
        self.assertEqual(self.call("POST", "/ask", raw=b"not json")[0], 400)
        self.assertEqual(self.call("POST", "/ask", raw=b"[1]")[0], 400)

    def test_llm_errors_map_to_status_codes(self):
        self.start(LLMConnectionError("Ollama down"), LLMModelNotFound("pull it"))
        self.assertEqual(self.call("POST", "/ask", {"question": "q"}), (503, {"error": "Ollama down"}))
        self.assertEqual(self.call("POST", "/ask", {"question": "q"})[0], 503)

    def test_unexpected_error_is_500_without_details(self):
        self.start(RuntimeError("secret internals"))
        status, body = self.call("POST", "/ask", {"question": "q"})
        self.assertEqual(status, 500)
        self.assertNotIn("secret", body["error"])

    def test_browser_requests_are_refused(self):
        self.start("x")
        status, _ = self.call("POST", "/ask", {"question": "q"}, headers={"Origin": "https://evil.example"})
        self.assertEqual(status, 403)

    def test_dns_rebinding_host_is_refused(self):
        self.start("x")
        status, _ = self.call("GET", "/health", headers={"Host": "evil.example:8765"})
        self.assertEqual(status, 403)

    def test_requires_json_content_type(self):
        self.start("x")
        status, _ = self.call("POST", "/ask", {"question": "q"}, headers={"Content-Type": "text/plain"})
        self.assertEqual(status, 415)

    def test_token(self):
        self.start("ok", token="s3cret")
        self.assertEqual(self.call("GET", "/health")[0], 401)
        self.assertEqual(self.call("GET", "/health", headers={"X-Turbo-Token": "wrong"})[0], 401)
        self.assertEqual(self.call("POST", "/ask", {"question": "q"}, headers={"X-Turbo-Token": "s3cret"})[0], 200)

    def test_unknown_endpoint(self):
        self.start()
        self.assertEqual(self.call("GET", "/nope")[0], 404)


if __name__ == "__main__":
    unittest.main()
