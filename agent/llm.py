"""Thin wrapper over the Mistral chat API so the loop stays testable.

Exposes one method, `chat(messages, tools, json_mode)` returning a normalized
`LLMResponse(message, tool_calls, usage)`. The agent loop only depends on this
interface, so tests can inject a fake LLM client.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Literal

from mistralai.client import Mistral as MistralSDK


@dataclass
class ToolCallRequest:
    """A tool invocation requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    """Normalized chat response the agent loop consumes."""

    content: str | None = None
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str | None = None


class MistralLLM:
    """Minimal chat wrapper around the official Mistral SDK."""

    def __init__(self, api_key: str | None = None, model: str = "mistral-large-latest", timeout_ms: int = 60_000):
        self.api_key = api_key or os.environ.get("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY not set and no api_key given")
        self.model = model
        self.timeout_ms = timeout_ms
        self._client = MistralSDK(api_key=self.api_key)

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        json_mode: bool = False,
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
            content=msg.content,
            tool_calls=tool_calls,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            finish_reason=choice.finish_reason,
        )
