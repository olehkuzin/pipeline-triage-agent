"""The triage agent: an LLM tool-calling loop over the read-only MCP server.

Design:
- Tools are reached only through an MCP client session (`MCPToolClient`); the
  function-calling specs shown to the model come from the server's tool list.
- Guardrails: hard tool-call budget per incident (when exhausted, the model must
  answer with what it has); every evidence quote must appear verbatim in the
  output of the tool it names; `requires_human_approval` is always True; no
  tool can change anything.
- Structured output: the final answer is validated against the Pydantic
  Diagnosis model; invalid output earns a bounded number of repair turns, and
  the agent falls back to `unknown` rather than guessing.
- Tracing: every incident records tool calls, token usage, turns, wall time and
  why (if at all) it fell back.
"""

from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any

from .llm import LLMClient
from .prompts import SYSTEM_PROMPT, USER_TEMPLATE
from .schemas import SOURCE_TOOL, Diagnosis, EvidenceItem, Incident
from .tools import MCPToolClient

MAX_TOOL_CALLS = 12
MAX_TURNS = 10
MAX_REPAIR_ATTEMPTS = 2

BUDGET_EXHAUSTED_MSG = (
    "ERROR: tool-call budget of {budget} calls for this incident is exhausted; this call was not executed. "
    "Do not call any more tools. Respond now with ONLY the JSON diagnosis based on the evidence you already have."
)
REPAIR_INSTRUCTIONS = (
    "Tools are no longer available; do not try to call them. Reply with ONLY the JSON diagnosis object, "
    "with exactly these keys: run_id, root_cause, confidence, evidence, proposed_next_step, "
    "requires_human_approval. Each evidence item needs a source (run_logs, table_schema, recent_changes or "
    "run_history) and a quote of at least 4 characters copied verbatim from that tool's output. "
    "An empty tool result such as [] is not evidence; leave it out."
)


@dataclass
class Trace:
    """Everything the agent did for one incident (for observability + eval)."""

    run_id: str
    turns: int = 0
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_call_count: int = 0
    budget_exhausted: bool = False
    repairs: int = 0
    rejections: list[str] = field(default_factory=list)  # why each final answer was rejected
    prompt_tokens: int = 0
    completion_tokens: int = 0
    wall_time_s: float = 0.0
    fallback_reason: str | None = None  # tool_budget_exceeded | invalid_output | max_turns_exceeded

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


class TriageAgent:
    def __init__(self, llm: LLMClient, tools: MCPToolClient, max_tool_calls: int = MAX_TOOL_CALLS):
        self.llm = llm
        self.tools = tools
        self.max_tool_calls = max_tool_calls
        self.last_trace: Trace | None = None

    # ---------- main loop ----------

    def diagnose(self, incident: Incident) -> Diagnosis:
        t0 = time.perf_counter()
        trace = Trace(run_id=incident.run_id)
        self.last_trace = trace
        outputs: dict[str, list[str]] = defaultdict(list)  # tool name -> successful outputs this incident
        specs = self.tools.function_specs()

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_TEMPLATE.format(
                    run_id=incident.run_id,
                    pipeline=incident.pipeline,
                    failed_task=incident.failed_task,
                    started_at=incident.started_at.isoformat(),
                ),
            },
        ]

        tools_allowed = True  # False after the budget runs out or an invalid answer: answer-only turns
        diagnosis: Diagnosis | None = None
        reason: str | None = None

        for _turn in range(MAX_TURNS):
            resp = self.llm.chat(
                messages,
                tools=specs,
                json_mode=not tools_allowed,
                tool_choice="auto" if tools_allowed else "none",
            )
            trace.turns += 1
            trace.prompt_tokens += resp.prompt_tokens
            trace.completion_tokens += resp.completion_tokens

            # --- tool-calling path ---
            if resp.tool_calls and tools_allowed:
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
                # every tool call must get a tool message, even the ones we refuse to run
                for tc in resp.tool_calls:
                    if trace.tool_call_count >= self.max_tool_calls:
                        trace.budget_exhausted = True
                        content = BUDGET_EXHAUSTED_MSG.format(budget=self.max_tool_calls)
                        trace.tool_calls.append({"name": tc.name, "arguments": tc.arguments, "skipped": True})
                    else:
                        trace.tool_call_count += 1
                        result = self.tools.call(tc.name, tc.arguments)
                        content = result.text
                        if not result.is_error:
                            outputs[tc.name].append(result.text)
                        trace.tool_calls.append(
                            {
                                "name": tc.name,
                                "arguments": tc.arguments,
                                "result_chars": len(result.text),
                                "is_error": result.is_error,
                            }
                        )
                    messages.append({"role": "tool", "tool_call_id": tc.id, "name": tc.name, "content": content})
                if trace.budget_exhausted:
                    tools_allowed = False
                continue

            # --- final-answer path ---
            candidate, error = self._parse_diagnosis(resp.content or "", incident.run_id)
            if candidate is not None:
                bad_quotes = self._check_evidence(candidate, outputs)
                if not bad_quotes:
                    diagnosis = candidate
                    break
                error = f"evidence quotes not found verbatim in the named tool's output: {bad_quotes}"
            trace.rejections.append(str(error))

            # --- repair: correct the model and retry in answer-only mode ---
            if trace.repairs >= MAX_REPAIR_ATTEMPTS:
                reason = "tool_budget_exceeded" if trace.budget_exhausted else "invalid_output"
                break
            trace.repairs += 1
            messages.append({"role": "assistant", "content": resp.content or ""})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Your previous answer was rejected: {error}\n\n" + REPAIR_INSTRUCTIONS
                    ),
                }
            )
            tools_allowed = False
        else:
            reason = "max_turns_exceeded"

        if diagnosis is None:
            trace.fallback_reason = reason or "invalid_output"
            diagnosis = self._fallback_diagnosis(incident.run_id, outputs)
        trace.wall_time_s = round(time.perf_counter() - t0, 2)
        return diagnosis

    # ---------- helpers ----------

    @staticmethod
    def _check_evidence(d: Diagnosis, outputs: dict[str, list[str]]) -> list[str]:
        """Quotes that do NOT appear verbatim in any successful output of the tool their `source` names."""
        bad: list[str] = []
        for ev in d.evidence:
            haystack = _norm("\n".join(outputs.get(SOURCE_TOOL[ev.source], [])))
            if _norm(ev.quote) not in haystack:
                bad.append(f"[{ev.source}] {ev.quote[:80]}")
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
        data["run_id"] = run_id  # the incident id is not the model's to change
        try:
            return Diagnosis.model_validate(data), None
        except Exception as e:  # pydantic ValidationError
            return None, str(e)[:300]

    @staticmethod
    def _fallback_diagnosis(run_id: str, outputs: dict[str, list[str]]) -> Diagnosis:
        """`unknown`, citing the last ERROR line of the logs verbatim when the agent fetched them."""
        evidence: list[EvidenceItem] = []
        log_lines = [ln for out in outputs.get("get_run_logs", []) for ln in out.splitlines() if ln.strip()]
        errors = [ln for ln in log_lines if " ERROR " in ln]
        if errors or log_lines:
            evidence.append(
                EvidenceItem(
                    source="run_logs",
                    quote=(errors or log_lines)[-1],
                    note="Agent could not validate a diagnosis; fallback to unknown (advisory only)",
                )
            )
        return Diagnosis(
            run_id=run_id,
            root_cause="unknown",
            confidence=0.0,
            evidence=evidence,
            proposed_next_step="The agent failed to produce a validated diagnosis; triage this run manually.",
        )
