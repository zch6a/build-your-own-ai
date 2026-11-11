"""Chapter 8 demo: chapter 7's tools served over MCP, discovered and called by chapter 7's agent.

    uv run python chapters/08-mcp/run.py

Starts `python -m byoai.mcp` as a subprocess and talks to it over its stdin/stdout.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from byoai.agent import Agent
from byoai.corpus import knowledge_base
from byoai.mcp import MCPClient
from byoai.planner import TownPlanner, agent_tasks, is_correct, town_tools


def show(direction: str, message: dict) -> None:
    text = json.dumps(message)
    print(f"  {direction} {text[:150]}{'…' if len(text) > 150 else ''}")


documents, questions = knowledge_base()
tasks = agent_tasks(documents, questions, 200)

t0 = time.perf_counter()
with MCPClient([sys.executable, "-m", "byoai.mcp"]) as client:
    startup = time.perf_counter() - t0
    tools = client.tools()
    # ---- 1. The conversation on the wire ------------------------------------------------------------------
    print(f"started the server in {startup * 1000:.0f} ms; the handshake and discovery, as sent:\n")
    for direction, message in client.trace:
        show(direction, message)

    # ---- 2. One call, and a failing one -------------------------------------------------------------------
    print("\na tool call, and a tool error (a normal result with isError — the model gets to read it):\n")
    client.trace.clear()
    calculate = next(t for t in tools if t.name == "calculate")
    calculate.fn(expression="35400 + 18500")
    try:
        calculate.fn(expression="__import__('os').system('rm -rf /')")
    except RuntimeError:
        pass
    for direction, message in client.trace:
        show(direction, message)

    # ---- 3. The agent, over MCP ---------------------------------------------------------------------------
    local, remote = Agent(TownPlanner(), town_tools(documents)), Agent(TownPlanner(), tools)
    timings = {}
    answers = {}
    for name, agent in (("in-process tools", local), ("tools over MCP", remote)):
        t0 = time.perf_counter()
        runs = [agent.run(t["task"]) for t in tasks]
        timings[name] = (time.perf_counter() - t0, sum(len(r.calls) for r in runs))
        answers[name] = [r.answer for r in runs]
    print(f"\nchapter 7's agent on {len(tasks)} tasks, with the same planner:")
    for name, (seconds, calls) in timings.items():
        correct = sum(is_correct(a, t) for a, t in zip(answers[name], tasks))
        print(f"  {name:<18} {correct}/{len(tasks)} correct   {calls} tool calls   {seconds * 1000 / calls:.2f} ms per call")
    same = answers["in-process tools"] == answers["tools over MCP"]
    print(f"  identical answers: {same}; the process boundary costs "
          f"{(timings['tools over MCP'][0] - timings['in-process tools'][0]) * 1000 / timings['tools over MCP'][1]:.2f} ms per call")

root = Path(__file__).resolve().parents[2]
print(f"\nuse it from any MCP client, e.g. Claude Code:\n  claude mcp add towns -- uv run --directory {root} python -m byoai.mcp")
