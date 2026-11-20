"""Small local HTTP/JSON server so editors (VS Code) can use the assistant.

Endpoints (all JSON):
    GET  /health
    POST /ask      {question, session_id?, context?, language?}
    POST /explain  {code, filename?, language?, question?, context?, session_id?}
    POST /fix      {code, filename?, language?, instruction?, context?, session_id?, start_line?}
    POST /reset    {session_id?}

Security: listens on 127.0.0.1 only by default. Requests from web pages are
refused (browsers always send an Origin header; the VS Code extension does
not), the Host header must be local (blocks DNS rebinding), and if TURBO_TOKEN
is set every request must send it in the X-Turbo-Token header.
"""

from __future__ import annotations

import hmac
import json
import logging
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import __version__
from .assistant import Assistant, InputError
from .config import Settings
from .llm import LLMConnectionError, LLMError, LLMModelNotFound, LLMTimeout

MAX_BODY_BYTES = 1_000_000
LOCAL_HOSTS = {"127.0.0.1", "localhost", "[::1]"}

log = logging.getLogger("turbo.server")


def _status_for(error: Exception) -> HTTPStatus:
    if isinstance(error, InputError):
        return HTTPStatus.BAD_REQUEST
    if isinstance(error, (LLMConnectionError, LLMModelNotFound)):
        return HTTPStatus.SERVICE_UNAVAILABLE
    if isinstance(error, LLMTimeout):
        return HTTPStatus.GATEWAY_TIMEOUT
    if isinstance(error, LLMError):
        return HTTPStatus.BAD_GATEWAY
    return HTTPStatus.INTERNAL_SERVER_ERROR


def make_handler(assistant: Assistant, settings: Settings) -> type[BaseHTTPRequestHandler]:
    def opt_str(body: dict, key: str) -> str | None:
        value = body.get(key)
        if value is None:
            return None
        if not isinstance(value, str):
            raise InputError(f"'{key}' must be a string.")
        return value

    def health(_body: dict) -> dict:
        info = {
            "ok": True,
            "version": __version__,
            "chat_model": settings.chat_model,
            "code_model": settings.code_model,
        }
        try:
            installed = assistant.llm.list_models()
            info["ollama"] = True
            info["missing_models"] = [
                m for m in {settings.chat_model, settings.code_model}
                if m not in installed and f"{m}:latest" not in installed
            ]
        except (LLMError, AttributeError):
            info["ollama"] = False
            info["missing_models"] = []
        return info

    def ask(body: dict) -> dict:
        return {"answer": assistant.ask(
            opt_str(body, "question") or "",
            session_id=opt_str(body, "session_id") or "default",
            context=opt_str(body, "context"),
            language=opt_str(body, "language"),
        )}

    def explain(body: dict) -> dict:
        return {"answer": assistant.explain(
            opt_str(body, "code") or "",
            filename=opt_str(body, "filename"),
            language=opt_str(body, "language"),
            question=opt_str(body, "question"),
            context=opt_str(body, "context"),
            session_id=opt_str(body, "session_id") or "default",
        )}

    def fix(body: dict) -> dict:
        start_line = body.get("start_line", 1)
        if not isinstance(start_line, int) or isinstance(start_line, bool) or start_line < 1:
            raise InputError("'start_line' must be a positive whole number.")
        return assistant.fix(
            opt_str(body, "code") or "",
            filename=opt_str(body, "filename"),
            language=opt_str(body, "language"),
            instruction=opt_str(body, "instruction"),
            context=opt_str(body, "context"),
            session_id=opt_str(body, "session_id") or "default",
            start_line=start_line,
        ).to_dict()

    def reset(body: dict) -> dict:
        return {"cleared": assistant.reset(opt_str(body, "session_id") or "default")}

    get_routes = {"/health": health}
    post_routes = {"/ask": ask, "/explain": explain, "/fix": fix, "/reset": reset}

    class Handler(BaseHTTPRequestHandler):
        server_version = f"TURBO/{__version__}"
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            log.info("%s - %s", self.address_string(), fmt % args)

        body_read = False

        def send_json(self, status: int, payload: dict) -> None:
            data = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            if not self.body_read:
                # Rejected before reading the body: close instead of reusing
                # the connection, so leftover bytes can't corrupt the next request.
                self.send_header("Connection", "close")
                self.close_connection = True
            self.end_headers()
            self.wfile.write(data)

        def error(self, status: HTTPStatus, message: str) -> None:
            if not self.body_read:
                self.drain_body()
            self.send_json(status, {"error": message})

        def drain_body(self) -> None:
            # Unread request data makes Windows reset the socket, and the
            # client would see "connection reset" instead of our error.
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                return
            if 0 < length <= MAX_BODY_BYTES:
                self.rfile.read(length)

        def check_request(self) -> bool:
            if self.headers.get("Origin"):
                self.error(HTTPStatus.FORBIDDEN, "Browser requests are not allowed.")
                return False
            host = (self.headers.get("Host") or "").rsplit(":", 1)[0].lower()
            if settings.host in ("127.0.0.1", "localhost") and host not in LOCAL_HOSTS:
                self.error(HTTPStatus.FORBIDDEN, "Host not allowed.")
                return False
            if settings.token:
                sent = self.headers.get("X-Turbo-Token", "")
                if not hmac.compare_digest(sent.encode(), settings.token.encode()):
                    self.error(HTTPStatus.UNAUTHORIZED, "Missing or wrong X-Turbo-Token.")
                    return False
            return True

        def dispatch(self, route, body: dict) -> None:
            try:
                self.send_json(HTTPStatus.OK, route(body))
            except Exception as error:  # noqa: BLE001 - turn every failure into JSON
                status = _status_for(error)
                if status == HTTPStatus.INTERNAL_SERVER_ERROR:
                    log.error("Unhandled error:\n%s", traceback.format_exc())
                    message = "Internal error in TURBO server. See the server window for details."
                else:
                    message = str(error)
                self.error(status, message)

        def do_GET(self):
            self.body_read = not self.headers.get("Content-Length")
            if not self.check_request():
                return
            route = get_routes.get(self.path.split("?", 1)[0])
            if route is None:
                self.error(HTTPStatus.NOT_FOUND, f"Unknown endpoint {self.path}")
                return
            self.dispatch(route, {})

        def do_POST(self):
            self.body_read = False
            if not self.check_request():
                return
            route = post_routes.get(self.path.split("?", 1)[0])
            if route is None:
                self.error(HTTPStatus.NOT_FOUND, f"Unknown endpoint {self.path}")
                return
            if "application/json" not in (self.headers.get("Content-Type") or ""):
                self.error(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "Send Content-Type: application/json.")
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = -1
            if length < 0 or length > MAX_BODY_BYTES:
                self.error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "Request body too large.")
                return
            raw = self.rfile.read(length)
            self.body_read = True
            try:
                body = json.loads(raw or b"{}")
                if not isinstance(body, dict):
                    raise ValueError
            except ValueError:
                self.error(HTTPStatus.BAD_REQUEST, "Body must be a JSON object.")
                return
            self.dispatch(route, body)

    return Handler


def create_server(assistant: Assistant, settings: Settings) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((settings.host, settings.port), make_handler(assistant, settings))
    server.daemon_threads = True
    return server


def serve(settings: Settings | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    settings = settings or Settings.from_env()
    if settings.host not in ("127.0.0.1", "localhost") and not settings.token:
        raise ValueError(
            f"TURBO_HOST={settings.host} exposes the server to your network. "
            "Set TURBO_TOKEN too, or use 127.0.0.1."
        )
    assistant = Assistant.from_settings(settings)
    try:
        server = create_server(assistant, settings)
    except OSError as error:
        raise ValueError(
            f"Can't listen on {settings.host}:{settings.port} ({error}). "
            "Is TURBO already running? Or pick another TURBO_PORT."
        ) from error

    host, port = server.server_address[:2]
    print(f"TURBO server on http://{host}:{port}  (chat: {settings.chat_model}, code: {settings.code_model})")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nTURBO server stopped.")
    finally:
        server.server_close()
