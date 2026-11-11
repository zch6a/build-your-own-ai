import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from byoai.llm import LLMError, OpenAICompatible, ScriptedLLM


class FakeServer:
    """A local stand-in for a chat-completions endpoint: replies from a queue, records requests."""

    def __init__(self):
        self.replies: list[tuple[int, dict | str]] = []
        self.requests: list[dict] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append({"path": self.path, "auth": self.headers.get("Authorization"), "body": body})
                status, reply = outer.replies.pop(0)
                data = (reply if isinstance(reply, str) else json.dumps(reply)).encode()
                self.send_response(status)
                if status == 429:
                    self.send_header("Retry-After", "0")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"


@pytest.fixture
def server():
    s = FakeServer()
    yield s
    s.httpd.shutdown()


def reply(content="", tool_calls=None):
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {"choices": [{"message": message}]}


def test_sends_a_chat_completion_request(server):
    server.replies.append((200, reply("Hello.")))
    client = OpenAICompatible(server.url, "deepseek-chat", api_key="test-key")
    out = client.chat([{"role": "user", "content": "Hi"}])
    assert out == {"role": "assistant", "content": "Hello.", "tool_calls": []}
    req = server.requests[0]
    assert req["path"] == "/v1/chat/completions" and req["auth"] == "Bearer test-key"
    assert req["body"]["model"] == "deepseek-chat" and req["body"]["messages"][0]["content"] == "Hi"
    assert "tools" not in req["body"]


def test_tools_are_sent_and_tool_calls_parsed(server):
    calls = [
        {"id": "c1", "type": "function", "function": {"name": "add", "arguments": '{"a": 1, "b": 2}'}},
        {"id": "c2", "type": "function", "function": {"name": "add", "arguments": '{"a": 1,'}},
    ]
    server.replies.append((200, reply(tool_calls=calls)))
    tool = {"name": "add", "description": "Add.", "parameters": {"type": "object", "properties": {}}}
    out = OpenAICompatible(server.url, "m").chat([{"role": "user", "content": "1+2"}], tools=[tool])
    assert server.requests[0]["body"]["tools"] == [{"type": "function", "function": tool}]
    assert server.requests[0]["auth"] is None  # no key, no header
    assert out["tool_calls"][0] == {"id": "c1", "name": "add", "arguments": {"a": 1, "b": 2}, "raw_arguments": '{"a": 1, "b": 2}'}
    assert out["tool_calls"][1]["arguments"] is None  # broken JSON is reported, not raised


def test_reasoning_content_is_kept(server):
    message = {"role": "assistant", "content": "", "reasoning_content": "Think first.", "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]}
    server.replies.append((200, {"choices": [{"message": message}]}))
    out = OpenAICompatible(server.url, "m").chat([])
    assert out["reasoning_content"] == "Think first." and out["tool_calls"][0]["name"] == "f"


def test_retries_rate_limits_and_server_errors(server):
    server.replies += [(429, {"error": "slow down"}), (500, "oops"), (200, reply("ok"))]
    assert OpenAICompatible(server.url, "m", retries=2).chat([])["content"] == "ok"
    assert len(server.requests) == 3


def test_client_errors_are_not_retried(server):
    server.replies += [(401, {"error": "bad key"}), (200, reply("never"))]
    with pytest.raises(LLMError, match="HTTP 401.*bad key"):
        OpenAICompatible(server.url, "m").chat([])
    assert len(server.requests) == 1


def test_unexpected_shape_is_an_error(server):
    server.replies.append((200, {"nothing": "here"}))
    with pytest.raises(LLMError, match="unexpected response"):
        OpenAICompatible(server.url, "m").chat([])


def test_a_non_json_body_is_an_llm_error(server):
    server.replies.append((200, "<html>gateway</html>"))
    with pytest.raises(LLMError, match="not JSON"):
        OpenAICompatible(server.url, "m").chat([])


def test_temperature_can_be_left_out(server):
    server.replies.append((200, reply("ok")))
    OpenAICompatible(server.url, "m").chat([], temperature=None)
    assert "temperature" not in server.requests[0]["body"]


def test_unreachable_endpoint():
    with pytest.raises(LLMError, match="could not reach"):
        OpenAICompatible("http://127.0.0.1:9", "m", retries=0, timeout=2).chat([])


def test_from_env_is_off_unless_configured(monkeypatch):
    for var in ("BYOAI_LLM_BASE_URL", "BYOAI_LLM_MODEL", "BYOAI_LLM_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")  # a key alone must not switch anything on
    assert OpenAICompatible.from_env() is None
    monkeypatch.setenv("BYOAI_LLM_BASE_URL", "https://api.deepseek.com")
    with pytest.raises(LLMError, match="BYOAI_LLM_MODEL"):
        OpenAICompatible.from_env()
    monkeypatch.setenv("BYOAI_LLM_MODEL", "deepseek-chat")
    client = OpenAICompatible.from_env()
    assert client.url == "https://api.deepseek.com/chat/completions" and client.api_key == "test-key"


def test_scripted_llm():
    llm = ScriptedLLM(["one", {"tool_calls": [{"id": "x", "name": "f", "arguments": {}}]}])
    assert llm.chat([{"role": "user", "content": "a"}])["content"] == "one"
    assert llm.chat([])["tool_calls"][0]["name"] == "f"
    assert len(llm.requests) == 2
    with pytest.raises(LLMError):
        llm.chat([])
