"""Chapter 7 demo: an agent loop, traced and measured — and what keeps it working when the model errs.

    uv run python chapters/07-agent/run.py

The model is the offline `TownPlanner` unless BYOAI_LLM_BASE_URL / BYOAI_LLM_MODEL / BYOAI_LLM_API_KEY
are set (see byoai/llm.py), in which case part 4 also runs 12 tasks with that model.
"""

from __future__ import annotations

import sys
import json

from byoai.agent import Agent
from byoai.corpus import knowledge_base
from byoai.llm import OpenAICompatible
from byoai.planner import Noisy, TownPlanner, agent_tasks, is_correct, town_tools

# Print UTF-8 even when the output is piped or redirected (Windows would otherwise use its legacy code page).
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

documents, questions = knowledge_base()
tasks = agent_tasks(documents, questions, 200)
tools = town_tools(documents)

# ---- 1. One run, step by step -------------------------------------------------------------------------------
task = next(t for t in tasks if t["kind"] == "total")
result = Agent(TownPlanner(), tools).run(task["task"])
print(f"task: {task['task']}\n")
for m in result.messages[2:]:
    if m["role"] == "assistant" and m.get("tool_calls"):
        for c in m["tool_calls"]:
            print(f"  model  → {c['function']['name']}({c['function']['arguments']})")
    elif m["role"] == "tool":
        content = m["content"]
        if content.startswith("["):
            content = " | ".join(f"{p['title']}: {p['text'][:50]}…" for p in json.loads(content))
        print(f"  tool   ← {content[:110]}")
    elif m["role"] == "assistant":
        print(f"  model  : {m['content']}")
print(f"\n  {result.steps} model turns, {len(result.calls)} tool calls; correct: {is_correct(result.answer, task)}")


# ---- 2. All tasks ---------------------------------------------------------------------------------------------
def measure(agent: Agent) -> dict:
    out = {}
    for kind in ("lookup", "compare", "total", "age"):
        group = [t for t in tasks if t["kind"] == kind]
        runs = [agent.run(t["task"]) for t in group]
        out[kind] = (sum(is_correct(r.answer, t) for t, r in zip(group, runs)) / len(group), sum(r.steps for r in runs) / len(runs))
    return out


print(f"\n{len(tasks)} tasks, success rate and model turns per task:")
print(f"  {'':<34}" + "".join(f"{k:>16}" for k in ("lookup", "compare", "total", "age")))
for name, planner in [("searches once", TownPlanner(tries=1)), ("+ other wording", TownPlanner(tries=2)),
                      ("+ whole article (default)", TownPlanner()), ("one call per turn", TownPlanner(parallel=False))]:
    m = measure(Agent(planner, tools))
    print(f"  {name:<34}" + "".join(f"{f'{ok:.0%} · {steps:.1f}':>16}" for ok, steps in m.values()))

# ---- 3. A model that makes mistakes ---------------------------------------------------------------------------
print("\na model that spoils a share of its tool calls (cut-off JSON, misspelled tool, wrong argument name):")
print(f"  {'bad calls':>10} {'errors fed back':>17} {'naive loop':>12}   turns (fed back)   hit step limit")
for p in (0.0, 0.1, 0.3, 0.5):
    rates, turns, limit = {}, 0.0, 0
    for feedback in (True, False):
        agent = Agent(Noisy(TownPlanner(), p, seed=1), tools, feedback=feedback)
        runs = [agent.run(t["task"]) for t in tasks]
        rates[feedback] = sum(is_correct(r.answer, t) for t, r in zip(tasks, runs)) / len(tasks)
        if feedback:
            turns, limit = sum(r.steps for r in runs) / len(runs), sum(r.stopped == "max_steps" for r in runs)
    print(f"  {p:>10.0%} {rates[True]:>17.0%} {rates[False]:>12.0%}   {turns:>16.1f}   {limit:>14}")

# ---- 4. A real model, if configured ---------------------------------------------------------------------------
llm = OpenAICompatible.from_env()
if llm is None:
    print("\nreal model: not configured (set BYOAI_LLM_BASE_URL, BYOAI_LLM_MODEL, BYOAI_LLM_API_KEY to try one)")
else:
    sample = tasks[:12]
    agent = Agent(llm, tools, system="Answer questions about towns. Use search_towns to find facts and calculate for arithmetic. Answer in one sentence.")
    runs = [agent.run(t["task"]) for t in sample]
    print(f"\n{llm.model}: {sum(is_correct(r.answer, t) for t, r in zip(sample, runs))}/{len(sample)} correct, "
          f"{sum(r.steps for r in runs) / len(runs):.1f} turns per task")
