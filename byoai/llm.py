"""An optional real model, for chapters 6–8: any OpenAI-compatible chat endpoint (DeepSeek included).

Everything in this project runs without it — the chapters use the chapter-3 model or deterministic
stand-ins, so the tests and demos need no key and cost nothing. If you have an endpoint, point these
environment variables at it and the RAG and agent demos will use it too:

    BYOAI_LLM_BASE_URL   e.g. https://api.deepseek.com   (nothing is sent anywhere unless this is set)
    BYOAI_LLM_MODEL      e.g. deepseek-chat
    BYOAI_LLM_API_KEY    your key (or DEEPSEEK_API_KEY / OPENAI_API_KEY)

The client is plain standard library: one POST to `/chat/completions`, retries on rate limits
and server errors, and tool calls parsed into plain dicts. It is what every SDK does underneath.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Iterable


class LLMError(RuntimeError):
    pass


class OpenAICompatible:
    """A chat-completions client: `chat(messages, tools=None) -> {"role", "content", "tool_calls"}`.

    `tool_calls` (chapter 7) is a list of `{"id", "name", "arguments"}` with `arguments` already parsed
    from JSON — or `None` when the model wrote JSON that does not parse, so the caller can tell it so.
    """

    def __init__(self, base_url: str, model: str, api_key: str | None = None, timeout: float = 60.0, retries: int = 2):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.api_key, self.timeout, self.retries = model, api_key, timeout, retries

    @classmethod
    def from_env(cls) -> OpenAICompatible | None:
        """A client from the environment variables above, or None if `BYOAI_LLM_BASE_URL` is not set."""
        base = os.environ.get("BYOAI_LLM_BASE_URL")
        if not base:
            return None
        model = os.environ.get("BYOAI_LLM_MODEL")
        if not model:
            raise LLMError("BYOAI_LLM_BASE_URL is set but BYOAI_LLM_MODEL is not")
        key = os.environ.get("BYOAI_LLM_API_KEY") or os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
        return cls(base, model, key)

    def chat(self, messages: list[dict], tools: list[dict] | None = None, temperature: float = 0.0) -> dict:
        body: dict = {"model": self.model, "messages": messages, "temperature": temperature}
        if tools:
            body["tools"] = [{"type": "function", "function": t} for t in tools]
        data = self._post(body)
        try:
            message = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"unexpected response shape: {str(data)[:200]}") from e
        calls = []
        for c in message.get("tool_calls") or []:
            fn = c.get("function", {})
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = None
            calls.append({"id": c.get("id", ""), "name": fn.get("name", ""), "arguments": args, "raw_arguments": fn.get("arguments", "")})
        return {"role": "assistant", "content": message.get("content") or "", "tool_calls": calls}

    def _post(self, body: dict) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = json.dumps(body).encode()
        for attempt in range(self.retries + 1):
            request = urllib.request.Request(self.url, data=payload, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read())
            except urllib.error.HTTPError as e:
                detail = e.read()[:300].decode(errors="replace")
                # 429 (rate limit) and 5xx are worth another try; 4xx (bad key, bad request) are not.
                if (e.code == 429 or e.code >= 500) and attempt < self.retries:
                    time.sleep(_retry_delay(e.headers.get("Retry-After"), attempt))
                    continue
                raise LLMError(f"HTTP {e.code} from {self.url}: {detail}") from None
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt < self.retries:
                    time.sleep(_retry_delay(None, attempt))
                    continue
                raise LLMError(f"could not reach {self.url}: {e}") from None
        raise AssertionError("unreachable")


def _retry_delay(retry_after: str | None, attempt: int) -> float:
    try:
        return min(float(retry_after), 30.0) if retry_after else 0.5 * 2**attempt
    except ValueError:
        return 0.5 * 2**attempt


class ScriptedLLM:
    """A model that replies from a script — for tests and offline demos. Each reply is a string (the
    content) or a full reply dict (e.g. with `tool_calls`). Records every request in `.requests`."""

    def __init__(self, replies: Iterable[str | dict]):
        self.replies = list(replies)
        self.requests: list[dict] = []

    def chat(self, messages: list[dict], tools: list[dict] | None = None, temperature: float = 0.0) -> dict:
        self.requests.append({"messages": [dict(m) for m in messages], "tools": tools})
        if not self.replies:
            raise LLMError("ScriptedLLM ran out of replies")
        reply = self.replies.pop(0)
        if isinstance(reply, str):
            return {"role": "assistant", "content": reply, "tool_calls": []}
        return {"role": "assistant", "content": "", "tool_calls": [], **reply}
