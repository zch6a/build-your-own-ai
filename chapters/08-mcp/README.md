# Chapter 8 — MCP: tools for any agent, over stdin and stdout

> **Code:** [`byoai/mcp.py`](../../byoai/mcp.py) (server, client and the town server, about 250 lines) · **Tests:** [`tests/test_mcp.py`](../../tests/test_mcp.py) · **Run:** `uv run python chapters/08-mcp/run.py` (two seconds)
>
> [← Chapter 7](../07-agent/README.md) · [All chapters](../../README.md#chapters)

Chapter 7's tools live inside the agent: to give a different agent the same tools, you would copy the
code. The Model Context Protocol (MCP) takes them out. A **server** is any program that offers tools;
a **client** — Claude Code, an IDE, chapter 7's agent — starts it, asks what it offers, and calls it.
Write a server once and every client that speaks the protocol can use it, in any language on either
side. This chapter builds both ends with the standard library, and connects them to the agent.

## The wire: JSON-RPC 2.0, one message per line

The client starts the server as a subprocess and writes to its stdin; the server answers on its stdout.
Each message is one line of JSON. A **request** has an `id` and gets exactly one response with the
same `id`; a **notification** has no `id` and gets none. The demo prints what is actually sent:

```text
→ {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": …}}
← {"jsonrpc": "2.0", "id": 1, "result": {"protocolVersion": "2025-06-18", "capabilities": {"tools": {"listChanged": false}}, "serverInfo": …}}
→ {"jsonrpc": "2.0", "method": "notifications/initialized"}
→ {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
← {"jsonrpc": "2.0", "id": 2, "result": {"tools": [{"name": "search_towns", "description": "Search the encyclopedia…", "inputSchema": …}]}}
→ {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "calculate", "arguments": {"expression": "35400 + 18500"}}}
← {"jsonrpc": "2.0", "id": 3, "result": {"content": [{"type": "text", "text": "53900"}], "isError": false}}
```

Three steps, then work:

1. **`initialize`** — the client says which protocol revision it speaks and what it can do; the server
   answers with the revision they will use and its own capabilities (here: tools, whose list does not
   change). If the server supports the client's revision it uses it; otherwise it offers its latest, and
   a client that cannot speak that one disconnects. Versioning is negotiated, never assumed.
2. **`notifications/initialized`** — the client is ready. A notification: no reply.
3. **`tools/list`** — names, descriptions and JSON Schemas: exactly chapter 7's `Tool.schema`, with
   `parameters` renamed to `inputSchema`. The client hands them to its model.

Then **`tools/call`** as often as the model likes.

One rule on the server side matters more than it looks: **stdout is the protocol**. A stray `print()`
in a tool would put a line on the wire that is not JSON-RPC and corrupt the conversation; logs go to
stderr. (On Windows the server also switches stdout to UTF-8 with `\n` line endings, which it would not
otherwise use.)

## Two kinds of failure

```text
→ {"…": "tools/call", "params": {"name": "calculate", "arguments": {"expression": "__import__('os').system('rm -rf /')"}}}
← {"…": "result": {"content": [{"type": "text", "text": "ValueError: not allowed in an arithmetic expression: Call"}], "isError": true}}
```

MCP separates who made the mistake:

* **Protocol errors** are JSON-RPC errors: unparseable JSON (`-32700`), a malformed request (`-32600`),
  an unknown method (`-32601`), an unknown tool (`-32602`). The *client program* got something wrong.
* **Tool errors** are normal results with `isError: true`: bad arguments, an exception in the tool. The
  *model* gets to read them and try again. (Revision 2025-06-18 still listed invalid arguments as a
  protocol error; 2025-11-25 moved them here, so the model can correct itself. This server follows the
  newer rule.)

That second kind is chapter 7's error feedback carried across a process boundary: the client turns an
`isError` result into an exception, and the agent turns the exception into an `error:` message for the
model, as it would for a local tool.

## The client

`MCPClient` starts the server, does the handshake, and turns each remote tool into a chapter-7 `Tool`
whose function sends `tools/call`. The agent cannot tell the difference — that is the point:

```python
with MCPClient(["uv", "run", "python", "-m", "byoai.mcp"]) as client:
    agent = Agent(model, client.tools())
    agent.run("How many people live in Brinuston and Nusgalgate together?")
```

The client has its own failure modes, each tested:

* **A server that hangs.** A reader thread moves every line from the server onto a queue, and each
  request waits on the queue with a timeout. A silent server is a `TimeoutError` after `timeout`
  seconds, not a frozen client. If it hangs during the handshake, the client kills it at once; a later
  request that times out is cancelled with `notifications/cancelled` (as the spec asks) and the server
  keeps running — it may only be slow, and a late reply is recognized by its `id` and dropped.
* **Requests from the server.** The server may send requests of its own (`ping`, and in the full
  protocol `sampling/createMessage` and others). JSON-RPC ids are per sender, so a server request can
  carry the same id as the client's pending one — a client that matched on `id` alone would take the
  server's `ping` for its tool result (the first version did; a review caught it). Anything with a
  `method` is a request or notification, never a reply: the client answers `ping` and refuses the rest.
* **A server that dies** closes its stdout; the reader posts a sentinel, and the waiting request
  fails with "server exited".
* **Noise on stdout.** Lines that are not JSON are skipped, and responses are matched by `id`, so a
  notification arriving in between does not get mistaken for the answer.
* **Shutdown**, in the spec's order: close the server's stdin; if it has not exited, SIGTERM (on
  Windows, TerminateProcess); if it still has not, SIGKILL. Always reaped — no zombie processes on
  Linux and macOS.
* **Pages.** `tools/list` may return a `nextCursor`; the client keeps asking until there is none, so a
  server with many tools does not silently lose the rest.

The server, for its part, must survive anything a client sends: `params` that are not an object, a
tool name that is not a string, and — if the server itself has a bug — an exception in `handle`, which
becomes an internal error (−32603) for that one request instead of ending the session. (The first
version crashed on `{"params": [1, 2]}`; three tests now send such messages.)

## What the demo shows

```text
chapter 7's agent on 199 tasks, with the same planner:
  in-process tools   199/199 correct   424 tool calls   0.05 ms per call
  tools over MCP     199/199 correct   424 tool calls   0.15 ms per call
  identical answers: True; the process boundary costs 0.10 ms per call
```

Same tasks, same planner, the tools in a separate process: identical answers, and a tenth of a
millisecond per call for two pipe writes, two JSON round trips and a thread hand-off. (Per-call times
include the planner's own work.) Starting the server — a Python process that imports numpy and builds
the BM25 index — takes about half a second, once. Next to a model call (hundreds of milliseconds to
seconds), the protocol is free.

## Use it from a real client

The town server is a real MCP server. Any MCP client can start it — for example Claude Code:

```bash
claude mcp add towns -- uv run --directory /path/to/build-your-own-ai python -m byoai.mcp
```

Add `--notes /some/folder` at the end to also offer `read_file`, sandboxed to that folder (chapter 7).
The tests check the message shapes against the protocol revision 2025-06-18 and run the server in a real
subprocess; it has not been tested against every client.

## Versions

MCP changes quickly. This chapter teaches the handshake-based revisions — 2024-11-05 to 2025-11-25,
all accepted here — because the handshake makes every step visible. Revision 2026-07-28 changes the
shape: no `initialize` or `ping`; the protocol version and capabilities travel in each request's
`_meta`; a new `server/discover` replaces the handshake; servers on stdio may no longer send requests
of their own; sampling, roots and logging are deprecated. Its stdio rules tell new clients to try
`server/discover` and fall back to `initialize` when a server answers "method not found" — which this
server does, so such clients can still use it. The tools themselves — names, schemas, `tools/call`,
`isError` — carry over unchanged.

## Exercises

1. Add `ping` handling to the client: send one every few seconds while idle, and treat a missing reply as
   a dead server. Careful: `request()` is not thread-safe — two threads reading the one reply queue
   each throw away the other's reply. Add a lock around it, or one reader that routes replies by `id`.
2. Implement `resources/list` and `resources/read` on the server, offering the 60 articles as resources
   with `town://Nustorford` URIs. What is the difference between a resource and a tool that returns
   the same text, from the model's point of view?
3. Send `notifications/tools/list_changed` from the server when a new tool is registered, and make the
   client re-fetch the list. Where must the client be careful (a call in flight)?
4. Write the server in another language — 60 lines of JavaScript reading lines from stdin — and run the
   tests' agent against it. Nothing in the client should change.
5. Make a tool that prints to stdout by mistake. What does the client see? How would you make the
   server robust to it?

## Where real systems differ

MCP also defines resources (data the client can read), prompts (templates the server offers), sampling
(the server asking the client's model for a completion), progress and cancellation for long calls, and
a second transport — streamable HTTP — for remote servers, with OAuth for authorization. Remote servers
bring the security questions of chapter 7 back at a larger scale: a server you install can describe
its tools in words meant to steer your model, so which servers you trust matters as much as which
tools you write.

---

That is the whole stack: bytes become tokens (1), gradients train a transformer (2, 3), it generates
text fast (4), vectors find related text (5), retrieval grounds answers in documents (6), a loop lets a
model use tools (7), and a protocol lets any program offer them (8). Every piece is small enough to
read in an evening and was measured before it was described.
