"""Chapter 8 — MCP: offer tools to any agent, over JSON-RPC on stdin/stdout.

Chapter 7's tools live inside the agent's process. The Model Context Protocol (MCP) moves them out: a
*server* is any program that offers tools; a *client* (Claude Code, an IDE, chapter 7's agent) starts
it, asks what it offers, and calls it. One server works with every client that speaks the protocol.

On the wire it is JSON-RPC 2.0, one JSON object per line, over the server's stdin and stdout:

    client → {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {...}}
    server ← {"jsonrpc": "2.0", "id": 1, "result": {"protocolVersion": ..., "capabilities": ...}}
    client → {"jsonrpc": "2.0", "method": "notifications/initialized"}          (no id: no reply)
    client → {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
    client → {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": ..., "arguments": ...}}

Run the town server (chapter 7's tools) and point any MCP client at it:

    uv run python -m byoai.mcp                      # search_towns, calculate
    uv run python -m byoai.mcp --notes ./notes      # + read_file, sandboxed to ./notes

Standard library only. This implements the tools part of the protocol (revision 2025-06-18); MCP also
defines resources, prompts and sampling, which follow the same request/response pattern.
"""

from __future__ import annotations

import json
import queue
import subprocess
import sys
import threading
from typing import IO

from byoai.agent import Tool, coerce, validate

PROTOCOL_VERSION = "2025-06-18"
SUPPORTED_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")

# JSON-RPC error codes
PARSE_ERROR, INVALID_REQUEST, METHOD_NOT_FOUND, INVALID_PARAMS, INTERNAL_ERROR = -32700, -32600, -32601, -32602, -32603


class MCPError(RuntimeError):
    def __init__(self, code: int, message: str):
        super().__init__(f"{message} (JSON-RPC error {code})")
        self.code = code


# ---- server -----------------------------------------------------------------------------------------------
class MCPServer:
    """Serves `tools` to one client over a pair of streams.

    Two kinds of failure, kept apart as the protocol asks:

    * **Protocol errors** — bad JSON, an unknown method, an unknown tool — are JSON-RPC errors: the
      client did something wrong.
    * **Tool errors** — arguments that do not fit the schema, an exception inside the tool — are a
      normal result with `isError: true`, so the *model* sees what went wrong and can try again
      (chapter 7's error feedback, across a process boundary).
    """

    def __init__(self, name: str, tools: list[Tool], version: str = "0.1.0"):
        self.info = {"name": name, "version": version}
        self.tools = {t.name: t for t in tools}
        self.initialized = False

    def handle(self, message: object) -> dict | None:
        """One incoming message → the response to send, or None (notifications get no reply).

        Nothing a client sends may crash the server: every shape is checked before it is used, and
        `serve` turns anything unexpected into an internal error (−32603) instead of exiting."""
        if isinstance(message, dict) and "method" not in message and ("result" in message or "error" in message):
            return None  # a response to a request of ours; this server sends none, so there is nothing to match
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0" or not isinstance(message.get("method"), str):
            return _error(message.get("id") if isinstance(message, dict) else None, INVALID_REQUEST, "not a JSON-RPC 2.0 request")
        method, params, msg_id = message["method"], message.get("params", {}), message.get("id")
        if not isinstance(params, dict):
            return None if "id" not in message else _error(msg_id, INVALID_PARAMS, "params must be an object")
        if "id" not in message:  # a notification
            if method == "notifications/initialized":
                self.initialized = True
            return None
        if method == "initialize":
            asked = params.get("protocolVersion")
            # Version negotiation: answer with the client's version if we support it, else our latest.
            version = asked if asked in SUPPORTED_VERSIONS else PROTOCOL_VERSION
            return _result(msg_id, {"protocolVersion": version, "capabilities": {"tools": {"listChanged": False}}, "serverInfo": self.info})
        if method == "ping":
            return _result(msg_id, {})
        if method == "tools/list":
            return _result(msg_id, {"tools": [{"name": t.name, "description": t.description, "inputSchema": t.parameters} for t in self.tools.values()]})
        if method == "tools/call":
            name, args = params.get("name"), params.get("arguments", {})
            if not isinstance(name, str) or name not in self.tools:
                return _error(msg_id, INVALID_PARAMS, f"unknown tool: {name!r}")
            problem = validate(args, self.tools[name].parameters)
            if problem:
                return _result(msg_id, _content(f"invalid arguments: {problem}", True))
            try:
                return _result(msg_id, _content(str(self.tools[name].fn(**coerce(args, self.tools[name].parameters))), False))
            except Exception as e:
                return _result(msg_id, _content(f"{type(e).__name__}: {e}", True))
        return _error(msg_id, METHOD_NOT_FOUND, f"method not found: {method}")

    def serve(self, stdin: IO[str] = sys.stdin, stdout: IO[str] = sys.stdout) -> None:
        """Read requests line by line until stdin closes. Only protocol messages go to stdout — a stray
        `print` would corrupt the stream, so anything else belongs on stderr."""
        for line in stdin:
            if not line.strip():
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError as e:
                response = _error(None, PARSE_ERROR, f"parse error: {e}")
            else:
                try:
                    response = self.handle(message)
                except Exception as e:  # a bug of ours must cost one request, not the whole session
                    msg_id = message.get("id") if isinstance(message, dict) else None
                    response = _error(msg_id, INTERNAL_ERROR, f"internal error: {type(e).__name__}: {e}")
            if response is not None:
                stdout.write(json.dumps(response) + "\n")
                stdout.flush()


def _result(msg_id, result: dict) -> dict:
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def _error(msg_id, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}


def _content(text: str, is_error: bool) -> dict:
    return {"content": [{"type": "text", "text": text}], "isError": is_error}


# ---- client -----------------------------------------------------------------------------------------------
class MCPClient:
    """Starts a server as a subprocess and talks to it.

        >>> with MCPClient(["uv", "run", "python", "-m", "byoai.mcp"]) as client:
        ...     tools = client.tools()          # chapter 7 `Tool`s that call the server
        ...     Agent(model, tools).run("...")

    A reader thread puts every line from the server on a queue, so a server that hangs or dies is a
    timeout or an error here, never a frozen client. `trace` records every message both ways.
    """

    def __init__(self, command: list[str], timeout: float = 30.0, client_name: str = "byoai"):
        self.timeout = timeout
        self.trace: list[tuple[str, dict]] = []
        self._next_id = 0
        self._lines: queue.Queue[str | None] = queue.Queue()
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                        text=True, encoding="utf-8", bufsize=1)
        threading.Thread(target=self._read, daemon=True).start()
        try:
            init = self.request("initialize", {"protocolVersion": PROTOCOL_VERSION, "capabilities": {}, "clientInfo": {"name": client_name, "version": "0.1.0"}})
        except BaseException:
            self.process.kill()  # do not leave a hung server running behind a failed start
            self.process.wait()
            raise
        self.server_info, self.protocol_version = init.get("serverInfo", {}), init.get("protocolVersion")
        if self.protocol_version not in SUPPORTED_VERSIONS:
            self.close()
            raise MCPError(INVALID_REQUEST, f"server speaks protocol {self.protocol_version!r}, which this client does not")
        self.notify("notifications/initialized")

    def _read(self) -> None:
        for line in self.process.stdout:
            self._lines.put(line)
        self._lines.put(None)  # the server closed its stdout (exited)

    def _send(self, message: dict) -> None:
        self.trace.append(("→", message))
        try:
            self.process.stdin.write(json.dumps(message) + "\n")
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            raise MCPError(INVALID_REQUEST, f"server is not running: {e}") from None

    def notify(self, method: str, params: dict | None = None) -> None:
        self._send({"jsonrpc": "2.0", "method": method, **({"params": params} if params else {})})

    def request(self, method: str, params: dict | None = None) -> dict:
        self._next_id += 1
        msg_id = self._next_id
        self._send({"jsonrpc": "2.0", "id": msg_id, "method": method, **({"params": params} if params else {})})
        while True:
            try:
                line = self._lines.get(timeout=self.timeout)
            except queue.Empty:
                raise TimeoutError(f"no reply to {method!r} within {self.timeout} s") from None
            if line is None:
                raise MCPError(INVALID_REQUEST, f"server exited while waiting for {method!r}")
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue  # not protocol output (a misbehaving server printed something); skip it
            self.trace.append(("←", message))
            if not isinstance(message, dict):
                continue
            if "method" in message:
                # The server's own request or notification. Ids are per sender, so a server request
                # may carry the same id as ours — it is not our answer. Answer ping; refuse the rest.
                if "id" in message:
                    reply = _result(message["id"], {}) if message["method"] == "ping" else _error(message["id"], METHOD_NOT_FOUND, f"client does not support {message['method']}")
                    self._send(reply)
                continue
            if message.get("id") != msg_id:
                continue  # a late reply to an earlier request that timed out
            if "error" in message:
                raise MCPError(message["error"].get("code", 0), message["error"].get("message", ""))
            return message.get("result", {})

    def tools(self) -> list[Tool]:
        """The server's tools as chapter-7 `Tool`s: calling one sends `tools/call`. A result with
        `isError` raises, so the agent feeds the message back to the model as an error."""
        def remote(name: str):
            def call(**arguments) -> str:
                result = self.request("tools/call", {"name": name, "arguments": arguments})
                text = "\n".join(c.get("text", "") for c in result.get("content", []) if c.get("type") == "text")
                if result.get("isError"):
                    raise RuntimeError(text)
                return text
            return call

        return [Tool(t["name"], t.get("description", ""), t.get("inputSchema", {"type": "object"}), remote(t["name"]))
                for t in self.request("tools/list").get("tools", [])]

    def close(self) -> None:
        """Ask the server to exit by closing its stdin; kill it if it has not within five seconds — and
        always reap it, so no zombie process is left behind on POSIX."""
        if self.process.poll() is None:
            try:
                self.process.stdin.close()  # stdin closing is the polite way to ask a stdio server to exit
            except OSError:
                pass
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()

    def __enter__(self) -> MCPClient:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# ---- the town server --------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    import argparse

    from byoai.agent import sandboxed_reader
    from byoai.corpus import knowledge_base
    from byoai.planner import town_tools

    parser = argparse.ArgumentParser(description="An MCP server (stdio) offering chapter 7's tools.")
    parser.add_argument("--notes", help="also offer read_file, limited to this folder")
    args = parser.parse_args(argv)
    documents, _ = knowledge_base()
    tools = town_tools(documents) + ([sandboxed_reader(args.notes)] if args.notes else [])
    # Windows would otherwise translate "\n" to "\r\n" and use the console's code page.
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8", newline="\n")
    MCPServer("byoai-towns", tools).serve()


if __name__ == "__main__":
    main()
