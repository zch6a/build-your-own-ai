"""An offline stand-in for the model in chapter 7, plus the tasks and tools it is tested on.

`TownPlanner` plays the part of the language model: it reads the conversation and replies with tool
calls or an answer, through the same `chat(messages, tools)` interface as a real endpoint. Its
planning is hand-written for one family of tasks (questions about the chapter-6 towns that need
lookups and arithmetic) — that is the part a real model does for any task. What it is good for is
making the *loop* observable and measurable: the agent, tools, validation and limits run exactly as
they would with a real model, and `OpenAICompatible` drops in unchanged.

Like a real model, it is stateless: every turn it re-reads the whole conversation to decide what to do.
"""

from __future__ import annotations

import json
import random
import re

from byoai.agent import Tool, safe_eval, tool
from byoai.rag import BM25, chunk_documents

FIELD = {"lookup": "mayor", "compare": "population", "total": "population", "age": "founded"}
# What to search for: the usual word, then the other wording an article may use, then — if the passage
# with the fact still did not rank in the top 3 — the town alone with more results (its whole article).
QUERIES = {
    "mayor": [("mayor", 3), ("governed", 3), ("", 10)],
    "population": [("population", 3), ("inhabitants people live", 3), ("", 10)],
    "founded": [("founded", 3), ("first houses built", 3), ("", 10)],
}
EXTRACT = {
    "mayor": re.compile(r"(?:mayor (?:of \w+ )?is|governed by) ([A-Z]\w+ [A-Z]\w+)"),
    "population": re.compile(r"[Aa]bout ([\d,]*\d)(?: people| inhabitants|\.)"),
    "founded": re.compile(r"(?:founded|built) in (\d{4})"),
}
YEAR = 2000


# ---- tasks ------------------------------------------------------------------------------------------------
def agent_tasks(documents: list[dict], questions: list[dict], n: int = 100, seed: int = 0) -> list[dict]:
    """Tasks of four kinds, `{"task", "answer", "kind"}`, with answers computed from the facts:

    * lookup:  "Who is the mayor of X?"                          — one search
    * compare: "Which has more inhabitants, X or Y?"             — two searches, a comparison
    * total:   "How many people live in X and Y together?"       — two searches, a calculation
    * age:     "How old was X in the year 2000?"                 — a search, a calculation
    """
    facts = {(q["doc"], q["field"]): q["answer"] for q in questions}
    r = random.Random(seed)
    tasks = []
    for i in range(n):
        kind = ["lookup", "compare", "total", "age"][i % 4]
        a, b = r.sample(range(len(documents)), 2)
        x, y = documents[a]["title"], documents[b]["title"]
        pop = lambda d: int(facts[(d, "population")].replace(",", ""))  # noqa: E731
        if kind == "lookup":
            tasks.append({"task": f"Who is the mayor of {x}?", "answer": facts[(a, "mayor")], "kind": kind})
        elif kind == "compare":
            if pop(a) == pop(b):
                continue
            tasks.append({"task": f"Which has more inhabitants, {x} or {y}?", "answer": x if pop(a) > pop(b) else y, "kind": kind})
        elif kind == "total":
            tasks.append({"task": f"How many people live in {x} and {y} together?", "answer": str(pop(a) + pop(b)), "kind": kind})
        else:
            tasks.append({"task": f"How old was {x} in the year {YEAR}?", "answer": str(YEAR - int(facts[(a, "founded")])), "kind": kind})
    return tasks


def is_correct(answer: str, expected: str) -> bool:
    """The expected name or number appears in the answer (numbers compared without commas)."""
    return expected.replace(",", "") in answer.replace(",", "")


# ---- tools ------------------------------------------------------------------------------------------------
def town_tools(documents: list[dict]) -> list[Tool]:
    """`search_towns` (chapter 6's BM25 over titled chunks) and `calculate` (`safe_eval`)."""
    chunks = chunk_documents(documents, max_words=40)
    bm25 = BM25([c.text for c in chunks])

    def search_towns(query: str, k: int = 3) -> str:
        """Search the encyclopedia of towns. Returns up to k passages as JSON, each with the article title.

        Args:
            query: keywords, e.g. "Nustorford mayor"
            k: how many passages to return (1-10)
        """
        hits = bm25.search(query, max(1, min(k, 10)))
        return json.dumps([{"title": chunks[i].title, "text": chunks[i].text.split(": ", 1)[1]} for i, _ in hits])

    def calculate(expression: str) -> str:
        """Evaluate an arithmetic expression with + - * / // % ** and parentheses.

        Args:
            expression: e.g. "25700 + 35300"
        """
        return str(safe_eval(expression))

    return [tool(search_towns), tool(calculate)]


# ---- the planner ------------------------------------------------------------------------------------------
def _parse_task(task: str) -> tuple[str, list[str]]:
    towns = re.findall(r"\b[A-Z][a-z]+\b", task)[1:]
    if task.startswith("Who is the mayor"):
        return "lookup", towns
    if task.startswith("Which has more"):
        return "compare", towns
    if task.startswith("How many people"):
        return "total", towns
    return "age", towns


class TownPlanner:
    """Replies like a model would: tool calls until it has the facts, then the answer.

    `parallel=True` asks for both towns' facts in one turn (several tool calls in one reply, as real
    models do); `False` asks one at a time. `tries` is how many of the `QUERIES` it will attempt per fact
    before saying it could not find it.
    """

    def __init__(self, parallel: bool = True, tries: int = 3):
        self.parallel, self.tries = parallel, tries

    def chat(self, messages: list[dict], tools: list[dict] | None = None, temperature: float = 0.0) -> dict:
        task = next(m["content"] for m in messages if m["role"] == "user")
        kind, towns = _parse_task(task)
        field = FIELD[kind]
        facts, searches, calc = self._read(messages, towns, field)
        missing = [t for t in towns if t not in facts]
        if missing:
            todo = [t for t in missing if searches[t] < min(self.tries, len(QUERIES[field]))]
            if not todo:
                return self._say(f"I could not find the {field} of {', '.join(missing)}.")
            todo = todo if self.parallel else todo[:1]
            return self._call(messages, [("search_towns", self._query(t, *QUERIES[field][searches[t]])) for t in todo])
        if kind == "lookup":
            return self._say(f"The mayor of {towns[0]} is {facts[towns[0]]}.")
        if kind == "compare":
            a, b = towns
            more = a if int(facts[a]) > int(facts[b]) else b
            return self._say(f"{more} has more inhabitants ({facts[a]} in {a}, {facts[b]} in {b}).")
        expression = f"{facts[towns[0]]} + {facts[towns[1]]}" if kind == "total" else f"{YEAR} - {facts[towns[0]]}"
        if calc.get(expression) is None:
            return self._call(messages, [("calculate", {"expression": expression})])
        if kind == "total":
            return self._say(f"{calc[expression]} people live in {towns[0]} and {towns[1]} together.")
        return self._say(f"{towns[0]} was {calc[expression]} years old in {YEAR}.")

    @staticmethod
    def _query(town: str, words: str, k: int) -> dict:
        return {"query": f"{town} {words}".strip()} if k == 3 else {"query": town, "k": k}

    @staticmethod
    def _read(messages: list[dict], towns: list[str], field: str):
        """Everything learned so far, from the conversation alone: facts per town, how many searches
        succeeded per town, and calculation results. Failed calls (results starting with "error:")
        teach nothing, so the planner simply tries again."""
        pending: dict[str, tuple[str, dict]] = {}
        facts: dict[str, str] = {}
        searches = {t: 0 for t in towns}
        calc: dict[str, str] = {}
        for m in messages:
            for c in m.get("tool_calls") or []:
                try:
                    pending[c["id"]] = (c["function"]["name"], json.loads(c["function"]["arguments"]))
                except (json.JSONDecodeError, TypeError):
                    pending[c["id"]] = (c["function"]["name"], {})
            if m["role"] != "tool" or m["content"].startswith("error:"):
                continue
            name, args = pending.get(m["tool_call_id"], ("", {}))
            if name == "search_towns":
                for t in towns:
                    if t in args.get("query", ""):
                        searches[t] += 1
                for passage in json.loads(m["content"]):
                    if passage["title"] in towns and passage["title"] not in facts:
                        found = EXTRACT[field].search(passage["text"])
                        if found:
                            facts[passage["title"]] = found.group(1).replace(",", "")
            elif name == "calculate":
                calc[args.get("expression", "")] = m["content"]
        return facts, searches, calc

    @staticmethod
    def _say(text: str) -> dict:
        return {"role": "assistant", "content": text, "tool_calls": []}

    @staticmethod
    def _call(messages: list[dict], calls: list[tuple[str, dict]]) -> dict:
        turn = len(messages)
        return {"role": "assistant", "content": "", "tool_calls": [
            {"id": f"call_{turn}_{i}", "name": name, "arguments": args, "raw_arguments": json.dumps(args)}
            for i, (name, args) in enumerate(calls)]}


class Noisy:
    """Wraps a model and spoils each tool call with probability `p` — the mistakes real models make:
    JSON cut off mid-way, a misspelled tool name, a wrong argument name."""

    def __init__(self, model, p: float, seed: int = 0):
        self.model, self.p, self.rng = model, p, random.Random(seed)

    def chat(self, messages: list[dict], tools: list[dict] | None = None, temperature: float = 0.0) -> dict:
        reply = self.model.chat(messages, tools, temperature)
        for call in reply.get("tool_calls") or []:
            if self.rng.random() >= self.p:
                continue
            kind = self.rng.choice(["truncated", "misspelled", "wrong_argument"])
            if kind == "truncated":
                call["raw_arguments"] = call["raw_arguments"][: len(call["raw_arguments"]) // 2]
                call["arguments"] = None
            elif kind == "misspelled":
                call["name"] = call["name"].replace("_", "") + "s"
            else:
                call["arguments"] = {"q" + k: v for k, v in call["arguments"].items()}
                call["raw_arguments"] = json.dumps(call["arguments"])
        return reply
