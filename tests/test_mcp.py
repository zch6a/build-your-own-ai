import io
import json
import sys
import time

import pytest

from byoai.agent import Agent, tool
from byoai.corpus import knowledge_base
from byoai.mcp import INVALID_PARAMS, INVALID_REQUEST, METHOD_NOT_FOUND, PARSE_ERROR, PROTOCOL_VERSION, SUPPORTED_VERSIONS, MCPClient, MCPError, MCPServer
from byoai.planner import TownPlanner, agent_tasks, is_correct, town_tools


@tool
def shout(text: str) -> str:
    """Upper-case the text."""
    if text == "fail":
        raise ValueError("refusing to shout that")
    return text.upper()


SERVER = MCPServer("test", [shout])


def req(method, params=None, id=1):
    return {"jsonrpc": "2.0", "id": id, "method": method, **({"params": params} if params is not None else {})}


# ---- the server, message by message ------------------------------------------------------------------------
def test_initialize_negotiates_the_version():
    r = SERVER.handle(req("initialize", {"protocolVersion": "2025-03-26", "capabilities": {}, "clientInfo": {"name": "t"}}))
    assert r["result"]["protocolVersion"] == "2025-03-26"  # a version we support: use the client's
    assert r["result"]["capabilities"] == {"tools": {"listChanged": False}} and r["result"]["serverInfo"]["name"] == "test"
    r = SERVER.handle(req("initialize", {"protocolVersion": "1999-01-01"}))
    assert r["result"]["protocolVersion"] == SUPPORTED_VERSIONS[0] == "2025-11-25"  # one we do not: offer our latest


def test_notifications_get_no_reply():
    server = MCPServer("t", [])
    assert server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None and server.initialized


def test_tools_list():
    tools = SERVER.handle(req("tools/list"))["result"]["tools"]
    assert tools == [{"name": "shout", "description": "Upper-case the text.", "inputSchema": shout.parameters}]


def test_tools_call_results_and_tool_errors():
    ok = SERVER.handle(req("tools/call", {"name": "shout", "arguments": {"text": "hi"}}))["result"]
    assert ok == {"content": [{"type": "text", "text": "HI"}], "isError": False}
    failed = SERVER.handle(req("tools/call", {"name": "shout", "arguments": {"text": "fail"}}))["result"]
    assert failed["isError"] and failed["content"][0]["text"] == "ValueError: refusing to shout that"
    bad_args = SERVER.handle(req("tools/call", {"name": "shout", "arguments": {"txt": "hi"}}))["result"]
    assert bad_args["isError"] and "missing required argument(s): text" in bad_args["content"][0]["text"]


@pytest.mark.parametrize("message, code", [
    (req("tools/call", {"name": "whisper", "arguments": {}}), INVALID_PARAMS),
    (req("resources/list"), METHOD_NOT_FOUND),
    ({"id": 1, "method": "ping"}, INVALID_REQUEST),
    ("ping", INVALID_REQUEST),  # (a list is a batch: one error per element, see test_review_fixes)
])
def test_protocol_errors(message, code):
    assert SERVER.handle(message)["error"]["code"] == code


@pytest.mark.parametrize("message", [
    req("tools/call", [1, 2]),
    req("initialize", "x"),
    req("tools/call", {"name": ["shout"], "arguments": {}}),
    req("tools/call", {"name": "shout", "arguments": [1]}),
])
def test_malformed_params_are_errors_not_crashes(message):
    r = SERVER.handle(message)
    assert ("error" in r and r["error"]["code"] == INVALID_PARAMS) or r["result"]["isError"]


def test_serve_survives_anything():
    class Buggy(MCPServer):
        def handle(self, message):
            if message.get("method") == "explode":
                raise KeyError("oops")
            return super().handle(message)

    lines = [json.dumps(req("explode", id=1)), json.dumps(req("tools/call", [1], id=2)), json.dumps(req("ping", id=3))]
    out = io.StringIO()
    Buggy("t", []).serve(io.StringIO("\n".join(lines) + "\n"), out)
    replies = [json.loads(line) for line in out.getvalue().splitlines()]
    assert [r["id"] for r in replies] == [1, 2, 3]
    assert replies[0]["error"]["code"] == -32603 and replies[2]["result"] == {}


def test_responses_from_the_client_are_ignored():
    assert SERVER.handle({"jsonrpc": "2.0", "id": 5, "result": {}}) is None


def test_serve_over_streams():
    lines = [json.dumps(req("ping", id=7)), "", "{not json", json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"})]
    out = io.StringIO()
    MCPServer("t", []).serve(io.StringIO("\n".join(lines) + "\n"), out)
    replies = [json.loads(line) for line in out.getvalue().splitlines()]
    assert replies[0] == {"jsonrpc": "2.0", "id": 7, "result": {}}
    assert replies[1]["error"]["code"] == PARSE_ERROR and replies[1]["id"] is None
    assert len(replies) == 2  # the blank line and the notification produced nothing


# ---- a real server in a real subprocess ---------------------------------------------------------------------
@pytest.fixture(scope="module")
def client(tmp_path_factory):
    notes = tmp_path_factory.mktemp("notes")
    (notes / "todo.txt").write_text("visit Nustorford", encoding="utf-8")
    with MCPClient([sys.executable, "-m", "byoai.mcp", "--notes", str(notes)], timeout=60) as c:
        yield c


def test_handshake_and_discovery(client):
    assert client.server_info["name"] == "byoai-towns" and client.protocol_version == PROTOCOL_VERSION
    assert [t.name for t in client.tools()] == ["search_towns", "calculate", "read_file"]
    sent = [m for direction, m in client.trace if direction == "→"]
    assert sent[0]["method"] == "initialize" and sent[1] == {"jsonrpc": "2.0", "method": "notifications/initialized"}


def test_remote_tools_behave_like_local_ones(client):
    tools = {t.name: t for t in client.tools()}
    assert tools["calculate"].fn(expression="25,700 + 35,300") == "61000"
    assert tools["read_file"].fn(path="todo.txt") == "visit Nustorford"
    assert json.loads(tools["search_towns"].fn(query="Nustorford mayor"))[0]["title"] == "Nustorford"
    with pytest.raises(RuntimeError, match="outside the notes folder"):
        tools["read_file"].fn(path="../secret.txt")
    with pytest.raises(RuntimeError, match="not allowed"):
        tools["calculate"].fn(expression="__import__('os')")
    with pytest.raises(MCPError, match="unknown tool"):
        client.request("tools/call", {"name": "nope", "arguments": {}})


def test_an_agent_over_mcp_gets_the_same_answers(client):
    docs, questions = knowledge_base()
    tasks = agent_tasks(docs, questions, 20)
    local = Agent(TownPlanner(), town_tools(docs))
    remote = Agent(TownPlanner(), [t for t in client.tools() if t.name != "read_file"])
    for t in tasks:
        a, b = local.run(t["task"]), remote.run(t["task"])
        assert a.answer == b.answer and is_correct(b.answer, t)


TRICKY_SERVER = r'''
import json, sys
def send(m):
    sys.stdout.write(json.dumps(m) + "\n"); sys.stdout.flush()
for line in sys.stdin:
    m = json.loads(line)
    if "method" not in m:
        continue
    if m["method"] == "initialize":
        send({"jsonrpc": "2.0", "id": m["id"], "result": {"protocolVersion": "2025-06-18", "capabilities": {}, "serverInfo": {"name": "tricky"}}})
    elif m["method"] == "tools/call":
        # First a request of the server's own, with the *same id* as the client's request.
        send({"jsonrpc": "2.0", "id": m["id"], "method": "ping"})
        pong = json.loads(sys.stdin.readline())
        send({"jsonrpc": "2.0", "id": m["id"], "result": {"content": [{"type": "text", "text": "pong was " + json.dumps(pong.get("result"))}], "isError": False}})
    elif m["method"] == "tools/list":
        # Two pages, to check the client follows nextCursor.
        page2 = m.get("params", {}).get("cursor") == "p2"
        tools = [{"name": "second", "inputSchema": {"type": "object"}}] if page2 else [{"name": "first", "inputSchema": {"type": "object"}}]
        send({"jsonrpc": "2.0", "id": m["id"], "result": {"tools": tools, **({} if page2 else {"nextCursor": "p2"})}})
    elif m["method"] == "hang":
        sys.stderr.write("hanging\n")  # never answer
    elif m["method"] == "notifications/cancelled":
        pass
'''


def test_server_requests_with_a_colliding_id_are_not_mistaken_for_replies():
    with MCPClient([sys.executable, "-c", TRICKY_SERVER], timeout=30) as client:
        result = client.request("tools/call", {"name": "x", "arguments": {}})
        assert result["content"][0]["text"] == "pong was {}"


def test_paginated_tool_lists_are_followed():
    with MCPClient([sys.executable, "-c", TRICKY_SERVER], timeout=30) as client:
        assert [t.name for t in client.tools()] == ["first", "second"]


def test_a_timeout_after_the_handshake_and_close_reaps_the_server():
    client = MCPClient([sys.executable, "-c", TRICKY_SERVER], timeout=30)  # a slow machine may take a while to start Python
    client.timeout = 1
    with pytest.raises(TimeoutError, match="hang"):
        client.request("hang")
    sent = [m for direction, m in client.trace if direction == "→"]
    assert sent[-1] == {"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {"requestId": 2, "reason": "timeout"}}
    client.close()
    assert client.process.returncode is not None


def test_a_silent_server_is_a_timeout_not_a_hang():
    started = time.perf_counter()
    with pytest.raises(TimeoutError, match="initialize"):
        MCPClient([sys.executable, "-c", "import time; time.sleep(30)"], timeout=1)
    assert time.perf_counter() - started < 10  # and the hung server was killed, not waited for


def test_a_server_that_exits_is_an_error():
    with pytest.raises(MCPError, match="exited"):
        MCPClient([sys.executable, "-c", "print('hello, not json')"], timeout=10)
