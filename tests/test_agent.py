import json
import os

import pytest

from byoai.agent import Agent, safe_eval, sandboxed_reader, tool, validate
from byoai.corpus import knowledge_base
from byoai.llm import ScriptedLLM
from byoai.planner import Noisy, TownPlanner, agent_tasks, is_correct, town_tools


@tool
def add(a: int, b: int = 0) -> str:
    """Add two integers.

    Args:
        a: the first number
        b: the second number
    """
    return str(a + b)


@tool
def boom(x: str) -> str:
    """Always fails."""
    raise RuntimeError("disk on fire")


def call(name, args, id="c1", raw=None):
    return {"id": id, "name": name, "arguments": args, "raw_arguments": raw if raw is not None else json.dumps(args)}


# ---- tool schemas and validation -------------------------------------------------------------------------
def test_tool_schema_from_signature_and_docstring():
    assert add.schema == {
        "name": "add",
        "description": "Add two integers.",
        "parameters": {
            "type": "object",
            "properties": {"a": {"type": "integer", "description": "the first number"}, "b": {"type": "integer", "description": "the second number", "default": 0}},
            "required": ["a"],
            "additionalProperties": False,
        },
    }


@pytest.mark.parametrize("args, problem", [
    ({"a": 1}, None),
    ([1], "must be a JSON object"),
    ({}, "missing required argument(s): a"),
    ({"a": 1, "c": 2}, "unknown argument(s): c"),
    ({"a": "1"}, "'a' must be an integer"),
    ({"a": True}, "'a' must be an integer"),
    ({"a": 3.0}, None),
    ({"a": 3.5}, "'a' must be an integer"),
])
def test_validate(args, problem):
    result = validate(args, add.parameters)
    assert result is None if problem is None else problem in result


# ---- the loop ---------------------------------------------------------------------------------------------
def test_tool_call_then_answer():
    llm = ScriptedLLM([{"tool_calls": [call("add", {"a": 2, "b": 3})]}, "It is 5."])
    result = Agent(llm, [add]).run("2+3?")
    assert (result.answer, result.steps, result.stopped) == ("It is 5.", 2, "answered")
    assert result.calls[0].result == "5" and not result.calls[0].error
    # What the model saw on its second turn: the call in wire format, then the result.
    second = llm.requests[1]["messages"]
    assert second[2]["tool_calls"][0] == {"id": "c1", "type": "function", "function": {"name": "add", "arguments": '{"a": 2, "b": 3}'}}
    assert second[3] == {"role": "tool", "tool_call_id": "c1", "content": "5"}
    assert llm.requests[0]["tools"] == [add.schema]


@pytest.mark.parametrize("bad, expected", [
    (call("ad", {"a": 1}), "unknown tool 'ad'; available: add"),
    (call("add", None, raw='{"a": 1'), "not valid JSON"),
    (call("add", {"x": 1}), "missing required argument(s): a"),
])
def test_mistakes_go_back_to_the_model(bad, expected):
    llm = ScriptedLLM([{"tool_calls": [bad]}, {"tool_calls": [call("add", {"a": 1}, id="c2")]}, "1"])
    result = Agent(llm, [add]).run("?")
    assert result.answer == "1" and result.calls[0].error and expected in result.calls[0].result
    assert llm.requests[1]["messages"][-1]["content"].startswith("error:")


def test_exceptions_in_tools_are_reported_not_raised():
    llm = ScriptedLLM([{"tool_calls": [call("boom", {"x": "y"})]}, "The tool failed."])
    result = Agent(llm, [boom]).run("?")
    assert result.calls[0].result == "error: RuntimeError: disk on fire" and result.answer == "The tool failed."


def test_naive_loop_stops_at_the_first_mistake():
    llm = ScriptedLLM([{"tool_calls": [call("ad", {"a": 1})]}, "never asked"])
    result = Agent(llm, [add], feedback=False).run("?")
    assert result.stopped == "error" and len(llm.requests) == 1


def test_step_limit():
    llm = ScriptedLLM([{"tool_calls": [call("add", {"a": i}, id=f"c{i}")]} for i in range(10)])
    result = Agent(llm, [add], max_steps=3).run("?")
    assert result.stopped == "max_steps" and result.steps == 3 and len(llm.requests) == 3


def test_repeated_calls_are_not_rerun():
    runs = []

    @tool
    def count(n: int) -> str:
        """Count."""
        runs.append(n)
        return "ok"

    llm = ScriptedLLM([{"tool_calls": [call("count", {"n": 1})]}, {"tool_calls": [call("count", {"n": 1}, id="c2")]}, "done"])
    result = Agent(llm, [count]).run("?")
    assert runs == [1] and "already made exactly this call" in result.calls[1].result


def test_a_stuck_model_is_stopped_early():
    llm = ScriptedLLM([{"tool_calls": [call("add", {"a": 1}, id=f"c{i}")]} for i in range(10)])
    result = Agent(llm, [add], max_steps=8).run("?")
    assert result.stopped == "repeating" and result.steps == 3 and len(llm.requests) == 3


def test_long_results_are_cut():
    @tool
    def big() -> str:
        """Big."""
        return "x" * 5000

    llm = ScriptedLLM([{"tool_calls": [call("big", {})]}, "ok"])
    result = Agent(llm, [big], max_result_chars=100).run("?")
    assert result.calls[0].result.startswith("x" * 100 + "… [cut: 4900 more")


def test_several_calls_in_one_turn():
    llm = ScriptedLLM([{"tool_calls": [call("add", {"a": 1}, id="a"), call("add", {"a": 2}, id="b")]}, "3"])
    result = Agent(llm, [add]).run("?")
    assert result.steps == 2 and [c.result for c in result.calls] == ["1", "2"]


# ---- safe tools -------------------------------------------------------------------------------------------
def test_safe_eval_does_arithmetic():
    assert safe_eval("25,700 + 35,300") == 61000
    assert safe_eval("(2 + 3) * -4 / 2 ** 2") == -5
    assert safe_eval("2000 - 1375") == 625


@pytest.mark.parametrize("attack", [
    "__import__('os').system('echo hi')",
    "open('secret.txt').read()",
    "(1).__class__",
    "9 ** 9 ** 9",
    "[1, 2]",
    "x + 1",
    "1 +" + " 1 +" * 100 + " 1",
    "(1,2)",      # was silently 12
    "1,5 * 2",    # a European decimal: was silently 30
    "(-8) ** 0.5",  # complex
])
def test_safe_eval_refuses_everything_else(attack):
    with pytest.raises((ValueError, SyntaxError)):
        safe_eval(attack)


def test_sandboxed_reader(tmp_path):
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "todo.txt").write_text("buy milk", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("hunter2", encoding="utf-8")
    read = sandboxed_reader(notes)
    assert read.fn(path="todo.txt") == "buy milk"
    for escape in ("../secret.txt", str(tmp_path / "secret.txt"), "sub/../../secret.txt"):
        with pytest.raises(PermissionError):
            read.fn(path=escape)
    with pytest.raises(FileNotFoundError, match="files: todo.txt"):
        read.fn(path="nope.txt")
    assert read.schema["parameters"]["required"] == ["path"]


def test_sandbox_sees_through_symlinks(tmp_path):
    notes = tmp_path / "notes"
    notes.mkdir()
    (tmp_path / "secret.txt").write_text("hunter2", encoding="utf-8")
    try:
        os.symlink(tmp_path / "secret.txt", notes / "link.txt")
    except (OSError, NotImplementedError):
        pytest.skip("cannot create symlinks here (Windows without developer mode)")
    with pytest.raises(PermissionError):
        sandboxed_reader(notes).fn(path="link.txt")


# ---- grading ----------------------------------------------------------------------------------------------
COMPARE = {"answer": "Kelby", "kind": "compare", "options": ["Kelby", "Morton"]}


@pytest.mark.parametrize("answer, task, ok", [
    ("Kelby has more inhabitants (5000 in Kelby, 300 in Morton).", COMPARE, True),
    ("Morton has more inhabitants (300 in Morton, 5000 in Kelby).", COMPARE, False),
    ("Between Morton and Kelby, Kelby has more people.", COMPARE, True),
    ("I could not find the population of Kelby, Morton.", COMPARE, False),
    ("31000 people live there.", {"answer": "1000"}, False),
    ("61,000 people live there.", {"answer": "61000"}, True),
    ("I don't know who the mayor of Kelby is; maybe Ada Venn.", {"answer": "Ada Venn"}, False),
])
def test_grading_is_strict(answer, task, ok):
    assert is_correct(answer, task) is ok


def test_a_planner_that_never_searches_scores_nothing():
    docs, questions = knowledge_base()
    tasks = agent_tasks(docs, questions, 40)
    agent = Agent(TownPlanner(tries=0), town_tools(docs))
    assert sum(is_correct(agent.run(t["task"]).answer, t) for t in tasks) == 0


# ---- the town agent ---------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def town():
    docs, questions = knowledge_base()
    return agent_tasks(docs, questions, 80), town_tools(docs)


def test_planner_solves_every_kind_of_task(town):
    tasks, tools = town
    agent = Agent(TownPlanner(), tools)
    results = [agent.run(t["task"]) for t in tasks]
    assert {t["kind"] for t in tasks} == {"lookup", "compare", "total", "age"}
    assert all(is_correct(r.answer, t) for t, r in zip(tasks, results))


def test_parallel_calls_save_turns(town):
    tasks, tools = town
    two_towns = [t for t in tasks if t["kind"] == "compare"]
    steps = {p: sum(Agent(TownPlanner(parallel=p), tools).run(t["task"]).steps for t in two_towns) for p in (True, False)}
    assert steps[True] < steps[False]


def test_searching_again_differently_matters(town):
    tasks, tools = town
    first_try_only = sum(is_correct(Agent(TownPlanner(tries=1), tools).run(t["task"]).answer, t) for t in tasks)
    assert first_try_only < 0.9 * len(tasks)


def test_error_feedback_recovers_from_a_noisy_model(town):
    tasks, tools = town
    rate = {}
    for feedback in (True, False):
        agent = Agent(Noisy(TownPlanner(), p=0.3, seed=1), tools, feedback=feedback)
        rate[feedback] = sum(is_correct(agent.run(t["task"]).answer, t) for t in tasks) / len(tasks)
    assert rate[True] > 0.95 and rate[False] < 0.7, rate
