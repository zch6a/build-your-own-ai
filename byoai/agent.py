"""Chapter 7 — an agent: a model that calls tools in a loop until the task is done.

An "agent" is less magic than it sounds. The model is given a list of tools (a name, a description and
a JSON Schema for the arguments). When it wants one, it replies with a *tool call* instead of text;
the program runs the tool and appends the result to the conversation; the model is asked again. When
it replies with text, that is the answer. Everything else — validation, errors, limits, sandboxing —
is what makes that loop safe to run.

    >>> from byoai.agent import Agent, tool
    >>> @tool
    ... def calculate(expression: str) -> str:
    ...     "Evaluate an arithmetic expression."
    ...     return str(safe_eval(expression))
    >>> agent = Agent(model, [calculate])
    >>> result = agent.run("What is 17 * 23?")
    >>> result.answer, result.steps

`model` is anything with `chat(messages, tools) -> {"content", "tool_calls"}`: a real endpoint through
`byoai.llm.OpenAICompatible`, a `ScriptedLLM`, or the offline `byoai.planner.TownPlanner`.
"""

from __future__ import annotations

import ast
import inspect
import json
import operator
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, get_type_hints

# ---- tools ------------------------------------------------------------------------------------------------
_JSON_TYPES = {str: "string", int: "integer", float: "number", bool: "boolean"}


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict  # JSON Schema of the arguments object
    fn: Callable[..., str]

    @property
    def schema(self) -> dict:
        """What the model is shown: the OpenAI / DeepSeek `function` format."""
        return {"name": self.name, "description": self.description, "parameters": self.parameters}


def tool(fn: Callable[..., str]) -> Tool:
    """Make a `Tool` from a typed function. The first paragraph of the docstring becomes the
    description; lines `name: text` under `Args:` describe the parameters. The model sees only this —
    a vague description is a tool the model will misuse."""
    hints = get_type_hints(fn)
    doc = inspect.getdoc(fn) or ""
    description = doc.split("\n\n")[0].replace("\n", " ").strip()
    arg_docs = dict(re.findall(r"^\s*(\w+):\s*(.+)$", doc.split("Args:", 1)[1], re.M)) if "Args:" in doc else {}
    properties, required = {}, []
    for name, param in inspect.signature(fn).parameters.items():
        prop = {"type": _JSON_TYPES[hints[name]]}
        if name in arg_docs:
            prop["description"] = arg_docs[name]
        if param.default is inspect.Parameter.empty:
            required.append(name)
        else:
            prop["default"] = param.default
        properties[name] = prop
    parameters = {"type": "object", "properties": properties, "required": required, "additionalProperties": False}
    return Tool(fn.__name__, description, parameters, fn)


def validate(args: object, schema: dict) -> str | None:
    """Check tool arguments against the (flat) schema; return what is wrong, in words the model can act
    on, or None. Models get arguments wrong more often than you would hope — the cure is telling them."""
    if not isinstance(args, dict):
        return "arguments must be a JSON object"
    props = schema.get("properties", {})
    missing = [r for r in schema.get("required", []) if r not in args]
    if missing:
        return f"missing required argument(s): {', '.join(missing)}"
    unknown = [a for a in args if a not in props]
    if unknown and schema.get("additionalProperties") is False:
        return f"unknown argument(s): {', '.join(unknown)}; expected: {', '.join(props)}"
    checks = {"string": str, "integer": int, "number": (int, float), "boolean": bool}
    for name, value in args.items():
        expected = props.get(name, {}).get("type")
        ok = isinstance(value, checks[expected]) if expected in checks else True
        if expected == "integer" and isinstance(value, float) and value.is_integer():
            ok = True  # JSON Schema: 3.0 is an integer (and models write it)
        if expected in ("integer", "number") and isinstance(value, bool):
            ok = False  # True is an int in Python, not a number in JSON
        if not ok:
            article = "an" if expected[0] in "aeiou" else "a"
            return f"argument {name!r} must be {article} {expected}, got {type(value).__name__}"
    return None


def coerce(args: dict, schema: dict) -> dict:
    """After `validate`: whole floats for integer parameters become ints, so the tool gets `3`, not `3.0`."""
    props = schema.get("properties", {})
    return {k: int(v) if props.get(k, {}).get("type") == "integer" and isinstance(v, float) else v for k, v in args.items()}


# ---- the loop ---------------------------------------------------------------------------------------------
@dataclass
class Step:
    """One tool call and what came back (for the trace)."""

    name: str
    arguments: object
    result: str
    error: bool


@dataclass
class Result:
    answer: str
    steps: int  # model turns
    calls: list[Step] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)
    stopped: str = "answered"  # or "max_steps", "repeating", "error"


class Agent:
    """The loop: ask the model; run the tools it calls; append the results; repeat until it answers.

    * Every failure — unknown tool, arguments that are not JSON, arguments that do not fit the
      schema, an exception inside the tool — goes back to the model as the tool's result, starting
      with "error:". The model can fix its call; a crashed loop cannot.
    * `max_steps` bounds the number of model turns: a confused model must not spin forever on your bill.
    * The same call repeated (same tool, same arguments) is answered with a reminder, not run again;
      after `max_repeats` such repeats the run stops — a model that keeps asking the same thing is stuck.
    * Tool results are cut to `max_result_chars`: whatever a tool returns is read by the model on every
      later turn.
    * `feedback=False` is the naive loop, for comparison: the first failed call ends the run.
    """

    def __init__(self, model, tools: list[Tool], system: str = "You are a helpful assistant. Use the tools when you need facts or arithmetic.", max_steps: int = 8, max_result_chars: int = 2000, feedback: bool = True, max_repeats: int = 2):
        self.model, self.system = model, system
        self.tools = {t.name: t for t in tools}
        self.max_steps, self.max_result_chars, self.feedback = max_steps, max_result_chars, feedback
        self.max_repeats = max_repeats

    def run(self, task: str) -> Result:
        messages = [{"role": "system", "content": self.system}, {"role": "user", "content": task}]
        schemas = [t.schema for t in self.tools.values()]
        calls: list[Step] = []
        seen: set[str] = set()
        repeats = 0
        for step in range(1, self.max_steps + 1):
            reply = self.model.chat(messages, tools=schemas)
            if not reply.get("tool_calls"):
                messages.append({"role": "assistant", "content": reply.get("content", "")})
                return Result(reply.get("content", ""), step, calls, messages)
            # The assistant turn, in the wire format the API expects back.
            turn = {"role": "assistant", "content": reply.get("content") or None, "tool_calls": [
                {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c.get("raw_arguments") or json.dumps(c["arguments"])}}
                for c in reply["tool_calls"]]}
            if reply.get("reasoning_content"):
                turn["reasoning_content"] = reply["reasoning_content"]  # thinking models need it back (see byoai.llm)
            messages.append(turn)
            for call in reply["tool_calls"]:
                result, error = self._execute(call, seen)
                calls.append(Step(call["name"], call["arguments"], result, error))
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})
                if error and not self.feedback:
                    return Result(f"failed: {result}", step, calls, messages, "error")
                if result.startswith("error: you already made exactly this call"):
                    repeats += 1
                    if repeats >= self.max_repeats:
                        return Result("Stopped: the model kept repeating the same call.", step, calls, messages, "repeating")
        return Result("I could not finish within the step limit.", self.max_steps, calls, messages, "max_steps")

    def _execute(self, call: dict, seen: set[str]) -> tuple[str, bool]:
        name, args = call["name"], call["arguments"]
        if name not in self.tools:
            return f"error: unknown tool {name!r}; available: {', '.join(self.tools)}", True
        if args is None:
            return f"error: the arguments were not valid JSON: {call.get('raw_arguments', '')[:100]!r}", True
        problem = validate(args, self.tools[name].parameters)
        if problem:
            return f"error: {problem}", True
        args = coerce(args, self.tools[name].parameters)
        key = name + json.dumps(args, sort_keys=True)
        if key in seen:
            return "error: you already made exactly this call; its result is above. Use it, or try something different.", True
        seen.add(key)
        try:
            out = str(self.tools[name].fn(**args))
        except Exception as e:  # the model should see what went wrong, not the loop die
            return f"error: {type(e).__name__}: {e}", True
        if len(out) > self.max_result_chars:
            out = out[: self.max_result_chars] + f"… [cut: {len(out) - self.max_result_chars} more characters]"
        return out, False


# ---- two tools worth getting right ----------------------------------------------------------------------------
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos}


def safe_eval(expression: str) -> float | int:
    """Arithmetic only. Never `eval()` what a model wrote: `__import__('os').system(...)` is an
    "expression" too. This walks the syntax tree and allows numbers and + − × ÷ // % ** — with the
    exponent capped, since `9**9**9` is arithmetic that never finishes."""
    def walk(node: ast.AST):
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](walk(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            left, right = walk(node.left), walk(node.right)
            if isinstance(node.op, ast.Pow) and (abs(right) > 100 or abs(left) > 1e6):
                raise ValueError("exponent too large")
            return _OPS[type(node.op)](left, right)
        raise ValueError(f"not allowed in an arithmetic expression: {type(node).__name__}")

    if len(expression) > 200:
        raise ValueError("expression too long")
    # Only thousands separators are removed ("25,700"); any other comma is an error, not something to
    # delete: "(1,2)" is not 12, and "1,5" may be a European 1.5.
    expression = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", expression)
    if "," in expression:
        raise ValueError("unexpected comma (thousands separators like 25,700 are fine)")
    result = walk(ast.parse(expression, mode="eval"))
    if isinstance(result, complex):
        raise ValueError("the result is not a real number")
    return result


def sandboxed_reader(root: str | Path, max_bytes: int = 20_000) -> Tool:
    """A `read_file` tool that can only read inside `root`. The model chooses the path, so the path is
    untrusted: `../../.ssh/id_rsa` and absolute paths are refused by resolving the path (symlinks
    included) and checking it is still inside the root."""
    base = Path(root).resolve()

    def read_file(path: str) -> str:
        """Read a text file from the notes folder.

        Args:
            path: path relative to the notes folder, e.g. "todo.txt"
        """
        target = (base / path).resolve()
        if not target.is_relative_to(base):
            raise PermissionError(f"{path!r} is outside the notes folder")
        if not target.is_file():
            raise FileNotFoundError(f"no file {path!r}; files: {', '.join(sorted(p.name for p in base.iterdir()))}")
        return target.read_bytes()[:max_bytes].decode("utf-8", errors="replace")

    return tool(read_file)
