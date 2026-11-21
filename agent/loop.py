"""The triage agent: a tool-calling loop over the same tools the MCP server exposes.

Design:
- The agent calls the same read-only tool implementations the MCP server serves
  (single source of truth; the MCP stdio server is the production wiring).
- Guardrails: hard tool-call budget per incident; evidence quotes must appear
  verbatim in tool outputs; `requires_human_approval` is always True; no
  execution of any fix.
- Structured output: the final answer is validated against the Pydantic
  Diagnosis model; invalid output earns a bounded number of repair turns,
  and the agent falls back to `unknown` rather than guessing.
- Tracing: every incident records tool calls, token usage, turns, wall time.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from pipeline_triage.env import Environment
from pipeline_triage.mcp_server import ToolError, get_recent_changes, get_run_history, get_run_logs, get_table_schema

from .llm import LLMResponse, MistralLLM
from .prompts import SYSTEM_PROMPT, USER_TEMPLATE
from .schemas import Diagnosis, EvidenceItem

MAX_TOOL_CALLS = 12
MAX_TURNS = 8
MAX_REPAIR_ATTEMPTS = 2

TOOL_IMPLS = {
    "get_run_logs": get_run_logs,
    "get_recent_changes": get_recent_changes,
    "get_table_schema": get_table_schema,
    "get_run_history": get_run_history,
}

TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "get_run_logs",
            "description": "Get the logs of a pipeline run, including the failing task.",
            "parameters": {
                "type": "object",
                "properties": {
                    "run_id": {"type": "string", "description": "The run to inspect."},
                    "tail_lines": {"type": "integer", "default": 200, "minimum": 10, "maximum": 2000},
                },
                "required": ["run_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_recent_changes",
            "description": (
                "Get code/config changes for a pipeline within the last N hours (relative to its latest run). "
                "A recent meaningful change shortly before the failure is a strong bad_deploy signal."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pipeline": {"type": "string"},
                    "since_hours": {"type": "integer", "default": 48, "minimum": 1, "maximum": 2160},
                },
                "required": ["pipeline"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_table_schema",
            "description": (
                "Get the current schema of a table (version + change description). "
                "A version bump with a change description indicates upstream schema drift."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "table": {"type": "string", "description": "Fully qualified name, e.g. raw.orders"},
                },
                "required": ["table"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_run_history",
            "description": (
                "Get recent run outcomes (success/failed) for a pipeline. Repeated earlier failures point to a "
                "persistent cause; a long healthy streak broken right after a change points to bad_deploy."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pipeline": {"type": "string"},
                    "last_n": {"type": "integer", "default": 10, "minimum": 1, "maximum": 100},
                },
                "required": ["pipeline"],
            },
        },
    },
]


class ToolCallBudgetExceeded(Exception):
    pass


@dataclass
class Trace:
    """Everything the agent did for one incident (for observability + eval)."""

    run_id: str
    turns: int = 0
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_call_count: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    wall_time_s: float = 0.0
    fallback_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


class TriageAgent:
    def __init__(self, env: Environment, llm: MistralLLM, max_tool_calls: int = MAX_TOOL_CALLS):
        self.env = env
        self.llm = llm
        self.max_tool_calls = max_tool_calls
        self.last_trace: Trace | None = None

    # ---------- main loop ----------

    def diagnose(self, run_id: str) -> Diagnosis:
        run = self.env.get_run(run_id)
        if run is None:
            raise ToolError(f"Run '{run_id}' not found.")

        t0 = time.perf_counter()
        trace = Trace(run_id=run_id)
        tool_outputs: list[str] = []  # every tool result this incident has seen

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_TEMPLATE.format(
                    run_id=run_id,
                    pipeline=run.pipeline,
                    failed_task=run.failed_task,
                    started_at=run.started_at.isoformat(),
                ),
            },
        ]

        tool_calls_used = 0
        repairs_used = 0
        force_json = False  # set after an invalid answer; repair turns get no tools + json mode
        diagnosis: Diagnosis | None = None

        for _turn in range(MAX_TURNS):
            resp = self.llm.chat(messages, tools=None if force_json else TOOL_SPECS, json_mode=force_json)
            trace.turns += 1
            trace.prompt_tokens += resp.prompt_tokens
            trace.completion_tokens += resp.completion_tokens

            # --- tool-calling path (skipped on repair turns) ---
            if resp.tool_calls and not force_json:
                messages.append(
                    {
                        "role": "assistant",
                        "content": resp.content or "",
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                            }
                            for tc in resp.tool_calls
                        ],
                    }
                )
                for tc in resp.tool_calls:
                    tool_calls_used += 1
                    trace.tool_call_count += 1
                    if tool_calls_used > self.max_tool_calls:
                        trace.fallback_reason = "tool_budget_exceeded"
                        self.last_trace = trace
                        trace.wall_time_s = round(time.perf_counter() - t0, 2)
                        raise ToolCallBudgetExceeded(
                            f"Exceeded tool call budget ({self.max_tool_calls}) for {run_id}"
                        )
                    result = self._call_tool(tc.name, tc.arguments)
                    tool_outputs.append(result)
                    trace.tool_calls.append(
                        {
                            "name": tc.name,
                            "arguments": tc.arguments,
                            "result_chars": len(result),
                            "is_error": result.startswith("ERROR:"),
                        }
                    )
                    messages.append(
                        {"role": "tool", "tool_call_id": tc.id, "name": tc.name, "content": result}
                    )
                continue

            # --- final-answer path ---
            candidate, error = self._parse_diagnosis(resp.content or "", run_id)
            if candidate is not None:
                bad_quotes = self._check_evidence(candidate, tool_outputs)
                if not bad_quotes:
                    diagnosis = candidate
                    break
                error = f"evidence quotes not found verbatim in tool outputs: {bad_quotes}"

            # --- repair: correct the model and retry with stricter output mode ---
            repairs_used += 1
            if repairs_used > MAX_REPAIR_ATTEMPTS:
                break
            messages.append({"role": "assistant", "content": resp.content or ""})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Your previous answer was invalid: {error}. "
                        "Respond again with ONLY the JSON diagnosis object. Copy evidence quotes "
                        "verbatim from tool outputs you received earlier."
                    ),
                }
            )
            force_json = True

        trace.wall_time_s = round(time.perf_counter() - t0, 2)
        self.last_trace = trace

        if diagnosis is None:
            diagnosis = self._fallback_diagnosis(run_id, tool_outputs)
            trace.fallback_reason = trace.fallback_reason or "invalid_output"

        return diagnosis

    # ---------- helpers ----------

    def _call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        impl = TOOL_IMPLS.get(name)
        if impl is None:
            return f"ERROR: unknown tool '{name}'. Available: {sorted(TOOL_IMPLS)}"
        try:
            out = impl(self.env, **arguments)
            return out if isinstance(out, str) else json.dumps(out)
        except ToolError as e:
            return f"ERROR: {e}"
        except TypeError as e:
            return f"ERROR: invalid arguments for {name}: {e}"

    @staticmethod
    def _check_evidence(d: Diagnosis, tool_outputs: list[str]) -> list[str]:
        """Return evidence quotes that do NOT appear in any tool output (verbatim)."""
        haystack = _norm(" ===== ".join(tool_outputs))
        bad: list[str] = []
        for ev in d.evidence:
            if _norm(ev.quote) not in haystack:
                bad.append(ev.quote[:80])
        return bad

    @staticmethod
    def _parse_diagnosis(raw: str, run_id: str) -> tuple[Diagnosis | None, str | None]:
        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S).strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            return None, f"invalid JSON: {e}"
        if not isinstance(data, dict):
            return None, "output is not a JSON object"
        data.setdefault("run_id", run_id)
        try:
            return Diagnosis.model_validate(data), None
        except Exception as e:  # pydantic ValidationError
            return None, str(e)[:300]

    @staticmethod
    def _fallback_diagnosis(run_id: str, tool_outputs: list[str]) -> Diagnosis:
        return Diagnosis(
            run_id=run_id,
            root_cause="unknown",
            confidence=0.0,
            evidence=[
                EvidenceItem(
                    source="run_logs",
                    quote=(tool_outputs[0][:200] + ("..." if len(tool_outputs[0]) > 200 else ""))
                    if tool_outputs
                    else "No tool outputs were obtained",
                    note="Agent could not validate its diagnosis; fallback to unknown (advisory only)",
                )
            ],
            proposed_next_step=(
                "The agent failed to produce a validated diagnosis; triage this run manually."
            ),
        )
