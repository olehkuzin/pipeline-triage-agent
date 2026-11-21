"""Agent loop tests with a scripted fake LLM (no API calls)."""

import json

import pytest

from pipeline_triage.env import Environment
from pipeline_triage.models import RootCause

from agent.llm import LLMResponse, ToolCallRequest
from agent.loop import TOOL_IMPLS, TriageAgent, ToolCallBudgetExceeded
from agent.schemas import Diagnosis


class FakeLLM:
    """Scripted LLM: returns queued responses; records everything it's asked."""

    def __init__(self, script: list[LLMResponse]):
        self.script = list(script)
        self.calls: list[dict] = []

    def chat(self, messages, tools=None, json_mode=False):
        self.calls.append({"messages": [m["role"] for m in messages], "tools": tools, "json_mode": json_mode})
        return self.script.pop(0)


@pytest.fixture(scope="module")
def env():
    return Environment.generate(n=40, seed=11)


def _drift_run(env):
    return next(r for r in env.all_runs() if r.root_cause == RootCause.SCHEMA_DRIFT)


def _tool_response(env, name, args):
    return TOOL_IMPLS[name](env, **args)


def test_agent_happy_path_tools_then_diagnosis(env):
    run = _drift_run(env)
    logs_result = _tool_response(env, "get_run_logs", {"run_id": run.run_id, "tail_lines": 200})
    table = next(iter(run.schemas))
    schema_result = _tool_response(env, "get_table_schema", {"table": table})

    # Find a real quote from the schema output for evidence
    quote = json.loads(schema_result)["change_description"]

    final_json = json.dumps(
        {
            "run_id": run.run_id,
            "root_cause": "schema_drift",
            "confidence": 0.9,
            "evidence": [
                {"source": "run_logs", "quote": "Column", "note": "missing column error"},
                {"source": "table_schema", "quote": quote, "note": "schema changed before run"},
            ],
            "proposed_next_step": "Update the column mapping to the renamed column and re-run from the failing task.",
            "requires_human_approval": True,
        }
    )

    fake = FakeLLM(
        [
            LLMResponse(
                tool_calls=[
                    ToolCallRequest(id="c1", name="get_run_logs", arguments={"run_id": run.run_id}),
                ]
            ),
            LLMResponse(
                tool_calls=[
                    ToolCallRequest(id="c2", name="get_table_schema", arguments={"table": table}),
                ]
            ),
            LLMResponse(content=final_json),
        ]
    )
    agent = TriageAgent(env, fake)
    d = agent.diagnose(run.run_id)

    assert d.root_cause == "schema_drift"
    assert d.requires_human_approval is True  # guardrail never disabled
    tr = agent.last_trace
    assert tr.tool_call_count == 2
    assert tr.turns == 3
    # The final pass should have no tool_calls in the assistant message and tools were offered every turn
    assert fake.calls[0]["tools"] is not None


def test_agent_invalid_evidence_is_rejected_then_repaired(env):
    run = _drift_run(env)
    logs_result = _tool_response(env, "get_run_logs", {"run_id": run.run_id, "tail_lines": 200})
    good_json = {
        "run_id": run.run_id,
        "root_cause": "schema_drift",
        "confidence": 0.85,
        "evidence": [
            {"source": "run_logs", "quote": "Column", "note": "column error"}
        ],
        "proposed_next_step": "Fix the column mapping and re-run the task.",
    }
    bad_json = dict(good_json)
    bad_json["evidence"] = [{"source": "run_logs", "quote": "this text is nowhere", "note": "invented"}]

    fake = FakeLLM(
        [
            LLMResponse(
                tool_calls=[ToolCallRequest(id="c1", name="get_run_logs", arguments={"run_id": run.run_id})]
            ),
            LLMResponse(content=json.dumps(bad_json)),  # first answer: fabricated evidence
            LLMResponse(content=json.dumps(good_json)),  # repair: valid
        ]
    )
    agent = TriageAgent(env, fake)
    d = agent.diagnose(run.run_id)
    assert d.root_cause == "schema_drift"
    assert agent.last_trace.tool_call_count == 1
    # repair pass should have been requested with json_mode forced and no tools
    repair_calls = [c for c in fake.calls if c["json_mode"]]
    assert repair_calls and repair_calls[0]["tools"] is None


def test_agent_falls_back_to_unknown_on_garbage(env):
    run = _drift_run(env)
    fake = FakeLLM([LLMResponse(content="I think it broke; probably the data team did something."),
                    LLMResponse(content="still not json"),
                    LLMResponse(content="nope")])
    agent = TriageAgent(env, fake)
    d = agent.diagnose(run.run_id)
    assert d.root_cause == "unknown"
    assert d.confidence == 0.0
    assert agent.last_trace.fallback_reason == "invalid_output"


def test_tool_budget_enforced(env):
    run = _drift_run(env)
    # model keeps requesting tools forever
    endless = LLMResponse(
        tool_calls=[ToolCallRequest(id="cx", name="get_run_logs", arguments={"run_id": run.run_id})]
    )
    fake = FakeLLM([endless] * 20)
    agent = TriageAgent(env, fake, max_tool_calls=3)
    with pytest.raises(ToolCallBudgetExceeded):
        agent.diagnose(run.run_id)
    assert agent.last_trace.tool_call_count == 4


def test_unknown_tool_returns_error_string(env):
    run = _drift_run(env)
    fake = FakeLLM(
        [
            LLMResponse(tool_calls=[ToolCallRequest(id="c1", name="get_secret_logs", arguments={})]),
            LLMResponse(
                content=json.dumps(
                    {
                        "run_id": run.run_id,
                        "root_cause": "unknown",
                        "confidence": 0.1,
                        "evidence": [{"source": "run_logs", "quote": "ERROR:", "note": "tool errored"}],
                        "proposed_next_step": "Investigate manually; no valid tool output.",
                    }
                )
            ),
        ]
    )
    agent = TriageAgent(env, fake)
    d = agent.diagnose(run.run_id)
    assert d.root_cause == "unknown"
    assert agent.last_trace.tool_calls[0]["is_error"] is True


def test_diagnosis_schema_guardrails(d=Diagnosis):
    # requires_human_approval is forced True even if the model says False
    d2 = d.model_validate(
        {
            "run_id": "x",
            "root_cause": "unknown",
            "confidence": 0.5,
            "evidence": [{"source": "run_logs", "quote": "boom!"}],
            "proposed_next_step": "check the thing manually",
            "requires_human_approval": False,
        }
    )
    assert d2.requires_human_approval is True
    # confidence is rounded
    d3 = d.model_validate(
        {
            "run_id": "x",
            "root_cause": "unknown",
            "confidence": 0.123456,
            "evidence": [{"source": "run_logs", "quote": "boom!"}],
            "proposed_next_step": "check the thing manually",
        }
    )
    assert d3.confidence == 0.12
