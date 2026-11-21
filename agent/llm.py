"""Thin wrapper over the Mistral chat API so the loop stays testable.

Exposes one method, `chat(messages, tools, json_mode, tool_choice)` returning a
normalized `LLMResponse`. The agent loop only depends on this interface
(`LLMClient`), so tests can inject a fake LLM client.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from mistralai.client import Mistral as MistralSDK


@dataclass
class ToolCallRequest:
    """A tool invocation requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any]


ToolChoice = Literal["auto", "none"]


@dataclass
class LLMResponse:
    """Normalized chat response the agent loop consumes."""

    content: str | None = None
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str | None = None


class LLMClient(Protocol):
    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        json_mode: bool = False,
        tool_choice: ToolChoice = "auto",
    ) -> LLMResponse: ...


def _text(content: Any) -> str | None:
    """Message content as text; some models return a list of chunks instead of a string."""
    if content is None or isinstance(content, str):
        return content
    return "".join(getattr(chunk, "text", "") or "" for chunk in content)


class MistralLLM:
    """Minimal chat wrapper around the official Mistral SDK."""

    def __init__(self, api_key: str | None = None, model: str = "mistral-large-latest", timeout_ms: int = 120_000):
        self.api_key = api_key or os.environ.get("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY not set and no api_key given")
        self.model = model
        self.timeout_ms = timeout_ms
        from mistralai.client.utils.retries import BackoffStrategy, RetryConfig

        # retry timeouts, connection errors, rate limits and 5xx with exponential backoff (max ~3 min)
        retries = RetryConfig(
            "backoff",
            BackoffStrategy(initial_interval=1_000, max_interval=20_000, exponent=2.0, max_elapsed_time=180_000),
            retry_connection_errors=True,
            status_codes_override=["429", "500", "502", "503", "504"],
        )
        self._client = MistralSDK(api_key=self.api_key, retry_config=retries)

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        json_mode: bool = False,
        tool_choice: ToolChoice = "auto",
    ) -> LLMResponse:
        from mistralai.client.models.responseformat import ResponseFormat

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "timeout_ms": self.timeout_ms,
        }
        if tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t["function"]["name"],
                        "description": t["function"].get("description", ""),
                        "parameters": t["function"]["parameters"],
                    },
                }
                for t in tools
            ]
            # "none" keeps the tool history valid while forbidding new calls
            kwargs["tool_choice"] = tool_choice
        if json_mode:
            kwargs["response_format"] = ResponseFormat(type="json_object")

        response = self._client.chat.complete(**kwargs)
        choice = response.choices[0]
        msg = choice.message

        tool_calls: list[ToolCallRequest] = []
        for tc in (msg.tool_calls or []):
            fn = tc.function
            try:
                args = json.loads(fn.arguments) if fn.arguments else {}
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(
                ToolCallRequest(id=tc.id or f"call_{len(tool_calls)}", name=fn.name, arguments=args)
            )

        usage = response.usage
        return LLMResponse(
            content=_text(msg.content),
            tool_calls=tool_calls,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            finish_reason=choice.finish_reason,
        )
