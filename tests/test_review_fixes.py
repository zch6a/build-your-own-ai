"""The defects an independent review found in the second polishing cycle, each pinned by a test."""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest

from byoai.agent import Agent, Tool, safe_eval
from byoai.autograd import Tensor
from byoai.mcp import INVALID_REQUEST, MCPServer
from byoai.rag import words
from byoai.tokenizer import Tokenizer
from byoai.vectordb import FlatIndex, HNSWIndex


def numeric_grad(f, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    g = np.zeros_like(x)
    for i in np.ndindex(x.shape):
        up, down = x.copy(), x.copy()
        up[i] += eps
        down[i] -= eps
        g[i] = (f(up) - f(down)) / (2 * eps)
    return g


def test_backward_through_a_used_part_of_a_graph_raises():
    x = Tensor([1.0, 2.0], requires_grad=True)
    h = x * x
    h.sum().backward()
    with pytest.raises(RuntimeError, match="part of this graph"):
        (h * 3).sum().backward()  # silently gave dx = [10, 20] (h's old gradient sent again), not [8, 16]
    assert x.grad.tolist() == [2.0, 4.0]
    y = Tensor([1.0, 2.0], requires_grad=True)
    h = y * y
    (h.sum() + (h * 3).sum()).backward()  # two losses on one trunk: add them, one backward
    assert y.grad.tolist() == [8.0, 16.0]


@pytest.mark.parametrize("sa, sb", [((3,), (2, 3, 4)), ((3,), (5, 2, 3, 4)), ((3,), (3, 4)), ((2, 3), (3,)), ((3,), (3,))])
def test_matmul_with_a_vector_has_the_right_gradients(sa, sb):
    rng = np.random.default_rng(0)
    a = Tensor(rng.normal(size=sa), requires_grad=True)
    b = Tensor(rng.normal(size=sb), requires_grad=True)
    (a @ b).sum().backward()  # a vector times a stack of matrices gave twice a's gradient and b's of the wrong shape
    assert a.grad.shape == sa and b.grad.shape == sb
    assert np.allclose(a.grad, numeric_grad(lambda v: (v @ b.data).sum(), a.data), atol=1e-5)
    assert np.allclose(b.grad, numeric_grad(lambda v: (a.data @ v).sum(), b.data), atol=1e-5)


def test_ndarray_at_tensor():
    out = np.ones((2, 3)) @ Tensor(np.ones((3, 2)))
    assert isinstance(out, Tensor) and out.shape == (2, 2)


def test_the_mcp_server_survives_bytes_that_are_not_utf8():
    lines = b'\xff\n{"jsonrpc":"2.0","id":1,"method":"ping"}\n'
    p = subprocess.run([sys.executable, "-m", "byoai.mcp"], input=lines, capture_output=True, timeout=60)
    replies = [json.loads(line) for line in p.stdout.decode("utf-8").splitlines()]
    assert p.returncode == 0  # a UnicodeDecodeError ended the server, and the ping went unanswered
    assert replies[0]["error"]["code"] == -32700 and replies[1] == {"jsonrpc": "2.0", "id": 1, "result": {}}


def test_mcp_batches_and_request_ids():
    server = MCPServer("t", [])
    batch = server.handle([{"jsonrpc": "2.0", "id": 1, "method": "ping"}, {"jsonrpc": "2.0", "method": "notifications/initialized"}])
    assert batch == [{"jsonrpc": "2.0", "id": 1, "result": {}}]  # 2025-03-26 requires batches
    assert server.handle([{"jsonrpc": "2.0", "method": "notifications/initialized"}]) is None
    assert server.handle([])["error"]["code"] == INVALID_REQUEST
    assert [r["error"]["code"] for r in server.handle([1, 2])] == [INVALID_REQUEST, INVALID_REQUEST]
    assert server.handle([[{"jsonrpc": "2.0", "id": 1, "method": "ping"}]])[0]["error"]["code"] == INVALID_REQUEST
    for bad in (None, {"a": 1}, True, 1.5):
        r = server.handle({"jsonrpc": "2.0", "id": bad, "method": "ping"})
        assert r["error"]["code"] == INVALID_REQUEST and r["id"] is None
    assert server.handle({"jsonrpc": "2.0", "id": "x", "method": "ping"})["result"] == {}


def reachable(index: HNSWIndex) -> set[int]:
    seen, stack = {index.entry}, [index.entry]
    while stack:
        for nb in index.links[stack.pop()][0]:
            if nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return seen


def test_exact_copies_in_hnsw_are_all_found():
    rng = np.random.default_rng(0)
    same = np.tile(rng.normal(size=16), (200, 1))
    x = np.vstack([same, rng.normal(size=(300, 16))])
    index = HNSWIndex(16, M=4)
    index.add(x)
    graph = set(range(len(x))) - {c for cs in index.copies.values() for c in cs}
    assert reachable(index) == graph  # as graph nodes, copies left a third of the data unreachable
    ids, scores = index.search(same[0], k=10)
    assert len(ids) == 10 and all(i < 200 for i in ids) and np.allclose(scores, 1.0, atol=1e-5)
    ids, _ = index.search(same[0], k=200)
    assert sorted(ids) == list(range(200))
    index.delete(0)  # the first copy is deleted: the others are still found
    assert 0 not in index.search(same[0], k=10)[0] and len(index.search(same[0], k=10)[0]) == 10


def test_queries_as_a_row_and_vectors_that_are_not_numbers():
    x = np.random.default_rng(1).normal(size=(50, 8))
    for index in (FlatIndex(8), HNSWIndex(8)):
        index.add(x)
        assert index.search(x[3][None], k=1)[0][0] == 3  # (1, d), as `add` accepts
        with pytest.raises(ValueError, match="one query"):
            index.search(x[:2], k=1)
        with pytest.raises(ValueError, match="NaN"):
            index.add(np.full((1, 8), np.nan))


class ManyCalls:
    """A model that asks for 500 calls in one turn."""

    def chat(self, messages, tools=None):
        if messages[-1]["role"] == "tool":
            return {"content": "done"}
        return {"tool_calls": [{"id": str(i), "name": "echo", "arguments": {"text": "x"}} for i in range(500)]}


def test_the_agent_limits_tool_calls_per_run():
    echo = Tool("echo", "Echo.", {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}, lambda text: text)
    result = Agent(ManyCalls(), [echo], max_steps=2, max_repeats=10_000).run("go")
    assert result.stopped == "max_calls" and len(result.calls) == 32  # all 500 ran, and the run said "answered"
    tool_results = [m for m in result.messages if m["role"] == "tool"]
    assert len(tool_results) == 500  # every call of the turn has a result, as the API requires


def test_safe_eval_says_which_part_of_a_power_is_too_large():
    with pytest.raises(ValueError, match="base too large"):
        safe_eval("1234567**2")
    with pytest.raises(ValueError, match="exponent too large"):
        safe_eval("2**1000")


def test_streaming_decoder_keeps_characters_whole():
    tok = Tokenizer.train("ab " * 50, vocab_size=260)
    ids = tok.encode("é 日本 🙂")
    show = tok.stream_decoder()
    assert "".join(show(i) for i in ids) == "é 日本 🙂"  # decode([token]) per token printed U+FFFD


def test_bm25_words_in_any_script():
    assert words("Zoë met 王小明 in 1907, 25,700 people") == ["zoë", "met", "王小明", "in", "1907", "25700", "people"]
